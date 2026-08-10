"""Import wiring, CLI contracts and cross-track isolation. CPU only, no weights.

The procedure track puts `procedure_track/`, `segment_track/` and `orena_sft/` on one
`sys.path` and imports from all three. That is the arrangement most likely to break
silently -- a module resolving to the wrong track's copy would be discovered at hour 3
of a training run, not here.
"""

import subprocess
import sys
from pathlib import Path

PROC = Path(__file__).resolve().parents[1]
REPO = PROC.parent
PY = REPO / "venv3.12/bin/python"

sys.path.insert(0, str(PROC))
sys.path.insert(0, str(PROC.parent / "segment_track"))
sys.path.insert(0, str(PROC.parent / "orena_sft"))

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def run(script, *args):
    return subprocess.run([str(PY), str(PROC / script), *args], capture_output=True, text=True)


print("1. the two collators coexist and stay distinct")
import collate as segment_collate  # noqa: E402

import collate_procedure  # noqa: E402

check("segment collate resolves to segment_track",
      "segment_track" in segment_collate.__file__, segment_collate.__file__)
check("procedure collate resolves to procedure_track",
      "procedure_track" in collate_procedure.__file__, collate_procedure.__file__)
check("they share the assistant marker",
      collate_procedure.ASSISTANT_MARKER == segment_collate.ASSISTANT_MARKER)
check("but have different clip_inputs",
      collate_procedure.clip_inputs is not segment_collate.clip_inputs)

print("\n2. the two samplers stay distinct")
import clip_sampling  # noqa: E402  (segment)

import sampling  # noqa: E402  (procedure)

check("segment sampler is segment_track's", "segment_track" in clip_sampling.__file__)
check("procedure sampler is procedure_track's", "procedure_track" in sampling.__file__)
check("procedure canonical fps is 5.0", sampling.FPS == 5.0)
check("segment default frame count is unchanged at 80",
      clip_sampling.DEFAULT_N_FRAMES == 80, "the shipped segment model depends on this")

print("\n3. prompts: procedure is its own, the shipped arms are untouched")
from prompts import build_system_prompt as shared_prompt  # noqa: E402

from prompts_procedure import build_system_prompt as proc_prompt  # noqa: E402

p_proc = proc_prompt(style="direct")
check("procedure prompt differs from segment's",
      p_proc != shared_prompt(style="direct", track="segment"))
check("shared module still builds frame", len(shared_prompt(style="direct", track="frame")) > 500)
check("shared module still builds segment", len(shared_prompt(style="direct", track="segment")) > 500)
try:
    shared_prompt(track="procedure")
    check("shared module is untouched (no procedure arm added to it)", False,
          "orena_sft/prompts.py grew a procedure arm; plan.md §2.5 says it should not")
except ValueError:
    check("shared module is untouched (no procedure arm added to it)", True)

print("\n4. scripts import and --help cleanly")
for script in ("build_procedure_sft_dataset.py", "sft_train_qwen_procedure_ddp.py",
               "evaluate_procedure.py"):
    r = run(script, "--help")
    check(f"{script} --help", r.returncode == 0, (r.stderr or "")[-250:])

print("\n5. trainer defaults match the committed configuration (plan.md §2.2)")
from sft_train_qwen_procedure_ddp import build_parser  # noqa: E402

d = vars(build_parser().parse_args([]))
for key, want in [("model_id", "Qwen/Qwen3.5-9B"), ("lora_r", 8), ("lora_alpha", 16),
                  ("batch_size", 1), ("grad_accum", 16), ("lr", 1e-4), ("seed", 42),
                  ("eval_steps", 50), ("save_steps", 50),
                  ("no_gradient_checkpointing", False),
                  ("frame_size", "640x360"), ("wandb_project", "orena-procedure-sft")]:
    check(f"{key} == {want!r}", d[key] == want, f"got {d[key]!r}")
check("effective batch is 32 on 2 GPUs", d["batch_size"] * d["grad_accum"] * 2 == 32)

print("\n6. bad arg combinations are rejected before the model loads")
r = run("sft_train_qwen_procedure_ddp.py", "--save-steps", "100", "--eval-steps", "150")
check("save_steps not a multiple of eval_steps",
      r.returncode != 0 and "multiple" in (r.stderr or ""))
r = run("build_procedure_sft_dataset.py", "--num-frames", "767")
check("odd --num-frames rejected", r.returncode != 0 and "even" in (r.stderr or ""))
r = run("evaluate_procedure.py", "--mode", "score", "--num-shards", "2")
check("score + num-shards rejected", r.returncode != 0)

print("\n7. the export builder and the sampler agree on the default cap")
from build_procedure_sft_dataset import DEFAULT_N_MAX as builder_cap  # noqa: E402

check("builder cap == sampling.DEFAULT_N_MAX", builder_cap == sampling.DEFAULT_N_MAX,
      str(builder_cap))
check("cap is even", builder_cap % 2 == 0)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
