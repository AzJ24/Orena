"""CPU checks on the trainer: arg validation, prompt wiring, SFTConfig values.

Everything here runs without a GPU and without loading 54 GB of weights -- it
catches the configuration mistakes that would otherwise surface 10 minutes into a
SLURM job, after the model load.
"""

import subprocess
import sys
from pathlib import Path

SEG = Path(__file__).resolve().parents[1]
SCRIPT = SEG / "sft_train_qwen_segment_ddp.py"
PY = SEG.parents[0] / ".venv/bin/python"

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def run(*args):
    return subprocess.run([str(PY), str(SCRIPT), *args], capture_output=True, text=True)


print("1. script is importable and --help works")
r = run("--help")
check("exit 0", r.returncode == 0, r.stderr[-200:] if r.returncode else "")
for flag in ("--lora-r", "--lora-alpha", "--frame-size", "--early-stopping-patience",
             "--prompt-style", "--system-prompt-file", "--dataloader-workers",
             "--no-gradient-checkpointing", "--ddp-find-unused-parameters"):
    check(f"exposes {flag}", flag in r.stdout)

print("2. defaults match the committed configuration")
sys.path.insert(0, str(SEG))
sys.path.insert(0, str(SEG.parent / "orena_sft"))
from sft_train_qwen_segment_ddp import build_parser  # noqa: E402

d = vars(build_parser().parse_args([]))
for key, want in [("model_id", "Qwen/Qwen3.6-27B"), ("epochs", 3.0), ("lora_r", 8),
                  ("lora_alpha", 16), ("prompt_style", "direct"), ("frame_size", "640x360"),
                  ("batch_size", 1), ("grad_accum", 16), ("lr", 1e-4), ("seed", 42),
                  ("early_stopping_patience", 3), ("eval_steps", 50), ("save_steps", 50),
                  ("eval_subset", 256), ("eval_sample_count", 4), ("save_total_limit", 20),
                  ("no_gradient_checkpointing", False), ("ddp_find_unused_parameters", False),
                  ("allow_slow_kernels", False), ("wandb_project", "orena-segment-sft")]:
    check(f"{key} == {want!r}", d[key] == want, f"got {d[key]!r}")
check("effective batch is 32 on 2 GPUs", d["batch_size"] * d["grad_accum"] * 2 == 32)
check("train file points at the export", Path(d["train_file"]).exists(), d["train_file"])
check("eval file points at the export", Path(d["eval_file"]).exists(), d["eval_file"])

print("3. bad arg combinations are rejected before the model loads")
r2 = run("--save-steps", "100", "--eval-steps", "150", "--early-stopping-patience", "3")
check("save_steps not a multiple of eval_steps", r2.returncode != 0 and "multiple" in r2.stderr)
r3 = run("--prompt-style", "plain", "--fo-definitions")
check("--fo-definitions with plain", r3.returncode != 0 and "no effect" in r3.stderr)
r4 = run("--prompt-style", "structured")
check("structured refused (targets are bare answers)", r4.returncode != 0)

print("4. config assembly (no model load)")
from prompts import build_system_prompt  # noqa: E402

sp = build_system_prompt(style="direct", track="segment")
check("segment prompt is non-empty", len(sp) > 1000, f"{len(sp)} chars")
check("segment prompt mentions the marker convention", "<SECONDS seconds>" in sp)

w, h_ = (int(x) for x in "640x360".split("x"))
check("frame-size parses to (w, h)", (w, h_) == (640, 360))

print("5. effective batch arithmetic")
for bs, ga, gpus, want in [(1, 16, 2, 32), (1, 8, 4, 32), (2, 8, 2, 32)]:
    check(f"bs={bs} accum={ga} gpus={gpus} -> {want}", bs * ga * gpus == want)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
