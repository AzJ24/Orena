"""GRPO training for Qwen3.5-9B on the FOCUS frame track, direct prompt.

Reward IS the FOCUS Evaluator's own correctness, so the training signal and the
leaderboard metric can't diverge:
  * closed formats (binary/number/fo_class/percentage/time): deterministic
    parse+compare via `focus.data.formats` -- byte-identical to what
    `Evaluator.run()` does.
  * open formats (open_ended/multiple_choice/matching): LLM-as-judge via
    `focus.evaluation.judges.APIJudge`, using the SAME judge prompt the Evaluator
    uses (`build_judge_prompt`), backed by an OpenAI model.

Two levers, both booleans:
  --init-from-sft / --no-init-from-sft
      warm-start from an SFT LoRA checkpoint (merged into the base weights, then
      a FRESH GRPO LoRA on top) vs cold-start from base Qwen. With PEFT, TRL
      computes the KL reference by disabling the trainable adapter, so the
      reference policy is automatically the merged-SFT init (warm) or plain base
      (cold) -- exactly the anchor you want in each case.
  --use-kl / --no-use-kl
      GRPO KL penalty (beta) on/off. On = anchor to the reference above (curbs
      further drift / forgetting during RL). Off = beta 0.

Everything else (model class, LoRA target, direct prompt, frame resolution) is
shared with the SFT/eval scripts so the pipeline stays consistent.

Note on the rules: an OpenAI judge as the reward signal needs internet AT
TRAINING (fine -- only the inference container is offline) but is the exact
"closed model as a training resource" question flagged in description.md sec.4/8.
Disclose it, or use --skip-open-formats to train verifiable-only with no judge.

Launch (H200):
    OPENAI_API_KEY=... .venv/bin/python orena_grpo/grpo.py \
        --datasets heico lapchole --init-from-sft --use-kl \
        --sft-adapter-dir orena_sft/checkpoints/heico-only-9b-r16-direct
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import statistics

import torch
import wandb
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS
from focus.data.data_models import Request
from focus.data.formats import JUDGE_FORMATS, get_format_class
from focus.evaluation.judges import APIJudge

GRPO_DIR = Path(__file__).resolve().parent          # orena_grpo/ (this script + outputs)
ORENA_SFT_DIR = GRPO_DIR.parent / "orena_sft"       # sibling: shared helpers + SFT checkpoints

# build_frame_sft_dataset (frame_path/DEFAULT_ROOT_DIR) and prompts live in
# orena_sft/; import them as a package from the repo root rather than duplicating.
sys.path.insert(0, str(GRPO_DIR.parent))
from orena_sft.build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402
from orena_sft.prompts import build_system_prompt, extract_answer  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file (KEY=VALUE lines), dependency-free.

    Slurm compute nodes and (per the VS Code warning) terminals don't inject
    .env automatically, so the script loads it itself -- works everywhere the
    file is readable, including sbatch, since /home is shared. A variable
    already set in the real environment WINS (so --export=ALL,OPENAI_API_KEY=...
    still overrides the file).
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)  # real env takes precedence


def condition_question(question: str, procedure: str | None) -> str:
    """Prepend the stated procedure to the question (see base_model_eval.py)."""
    return f"Procedure type: {procedure}.\n{question}" if procedure else question


# ── dataset ──────────────────────────────────────────────────────────────────

REWARD_COLS = ["answer", "format", "procedure_type", "question",
               "start_time", "end_time", "qID", "videoID"]


def build_dataset(datasets, cfg, system_prompt, condition_procedure, skip_open):
    """Load frame-track TRAIN QA for each dataset straight from FocusDataset
    (so we get real Request/Reference: start_time/end_time for the judge, the
    typed answer, and _format), resolve each frame on disk, and return an HF
    Dataset whose rows carry the metadata the reward needs.

    Images are loaded lazily via `with_transform`: the stored rows hold only a
    path + metadata (KB scale), and each access builds the chat `prompt` with
    the PIL image inline -- the shape TRL's VLM GRPO reads images from.
    """
    rows = []
    for ds in datasets:
        base_fps = float(DATASET_BASE_FPS[ds])
        for req, ref in FocusDataset(ds, DatasetSplit.TRAIN, Track.FRAME):
            if skip_open and ref._format in JUDGE_FORMATS:
                continue
            p = frame_path(cfg, ds, base_fps, req.videoID, req.start_time)
            if not p.exists():
                continue
            rows.append({
                "image_path": str(p),
                "question": req.question,
                "answer": str(ref.answer),
                "format": ref._format,
                "procedure_type": req.procedure_type,
                "start_time": float(req.start_time),
                "end_time": float(req.end_time),
                "qID": req.qID,
                "videoID": req.videoID,
            })
    if not rows:
        raise SystemExit("No usable training rows (check --root-dir / frame extraction).")

    hfds = Dataset.from_list(rows)

    sys_turn = ([{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
                if system_prompt else [])

    def transform(batch):
        prompts = []
        for i in range(len(batch["image_path"])):
            q = (condition_question(batch["question"][i], batch["procedure_type"][i])
                 if condition_procedure else batch["question"][i])
            prompts.append(sys_turn + [{"role": "user", "content": [
                {"type": "image", "image": Image.open(batch["image_path"][i]).convert("RGB")},
                {"type": "text", "text": q},
            ]}])
        return {"prompt": prompts, **{c: batch[c] for c in REWARD_COLS}}

    return hfds.with_transform(transform)


# ── reward ───────────────────────────────────────────────────────────────────

def _completion_text(c) -> str:
    """TRL returns a conversational completion as [{'role':'assistant','content':...}]
    (or a bare string for text-only). Pull the assistant text either way."""
    if isinstance(c, list):
        return c[-1]["content"] if c else ""
    return c


class RolloutRecorder:
    """Holds the most recent generation batch's rollouts so a callback can print
    one group (question, GT, every completion + its reward) every N steps. The
    reward function fills it; the callback reads it. Decoupled this way because
    the reward fn has the data but not the authoritative global_step, and the
    callback has the step but not the data."""

    def __init__(self):
        self.sample: dict | None = None  # one qID group: question, gt, format, items

    def record(self, qids, questions, gts, fmts, preds, rewards):
        # group completions by qID (each unique prompt has G of them), then
        # surface the group with the most reward spread -- the informative case
        # (mixed right/wrong = where GRPO actually learns).
        groups: dict[str, dict] = {}
        for qid, q, gt, fmt, pred, r in zip(qids, questions, gts, fmts, preds, rewards):
            g = groups.setdefault(qid, {"question": q, "gt": gt, "format": fmt, "items": []})
            g["items"].append((pred, r))
        if not groups:
            return
        best = max(groups.values(),
                   key=lambda g: statistics.pstdev([r for _, r in g["items"]])
                   if len(g["items"]) > 1 else 0.0)
        self.sample = best


class RolloutInspectionCallback(TrainerCallback):
    """Every `every` optimizer steps, print the latest recorded group and log it
    to a wandb table. TRL already logs reward / reward_std / frac_reward_zero_std
    (the advantage-collapse metric) each step; this adds the human-readable view."""

    def __init__(self, recorder: RolloutRecorder, every: int = 10):
        self.recorder = recorder
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0:
            return
        s = self.recorder.sample
        if s is None:
            return
        items = s["items"]
        mean_r = sum(r for _, r in items) / len(items)
        std_r = statistics.pstdev([r for _, r in items]) if len(items) > 1 else 0.0

        bar = "-" * 78
        print(f"\n{bar}\n[step {state.global_step}] rollout group  "
              f"(format={s['format']}, group reward mean={mean_r:.2f} std={std_r:.2f})\n{bar}")
        print(f"Q : {s['question'][:150]}")
        print(f"GT: {s['gt']!r}")
        for pred, r in items:
            print(f"  [{r:.1f}] {pred[:100]!r}")
        print(bar, flush=True)

        if wandb.run is not None:
            table = wandb.Table(columns=["step", "format", "question", "gt", "completion", "reward"])
            for pred, r in items:
                table.add_data(state.global_step, s["format"], s["question"], s["gt"], pred, r)
            # No explicit step=: TRL's own logging has already advanced wandb's
            # step counter past global_step, and a smaller step is dropped.
            wandb.log({"rollouts": table, "rollout/group_reward_mean": mean_r,
                       "rollout/group_reward_std": std_r})


def make_reward_fn(judge: APIJudge | None, judge_workers: int = 8,
                   recorder: RolloutRecorder | None = None):
    """Reward = the Evaluator's per-question correctness (0.0/1.0).

    Closed formats are scored by the format class's own read()+compare() -- the
    same code path Evaluator.run() takes. Open formats go to `judge` (APIJudge,
    same prompt as the Evaluator), run concurrently since each is a blocking
    HTTP call. A parse/judge failure scores 0.0, never raises.
    """
    def focus_evaluator_reward(prompts, completions, completion_ids=None, **kw):
        preds = [extract_answer(_completion_text(c)) for c in completions]
        fmts, answers = kw["format"], kw["answer"]
        rewards: list[float | None] = [None] * len(preds)
        judge_jobs = []

        for i, (pred, fmt, gt) in enumerate(zip(preds, fmts, answers)):
            if fmt in JUDGE_FORMATS:
                if judge is None:
                    rewards[i] = 0.0
                    continue
                req = Request(qID=kw["qID"][i], videoID=kw["videoID"][i],
                              start_time=kw["start_time"][i], end_time=kw["end_time"][i],
                              procedure_type=kw["procedure_type"][i], question=kw["question"][i])
                judge_jobs.append((i, req, gt, pred))
            else:
                try:
                    f = get_format_class(fmt)()
                    rewards[i] = 1.0 if f.compare(f.read(gt), f.read(pred)) else 0.0
                except Exception:
                    rewards[i] = 0.0

        if judge_jobs:
            def run(job):
                i, req, gt, pred = job
                try:
                    return i, (1.0 if judge.judge(req, gt, pred) else 0.0)
                except Exception:
                    return i, 0.0
            with ThreadPoolExecutor(max_workers=judge_workers) as ex:
                for i, r in ex.map(run, judge_jobs):
                    rewards[i] = r

        if recorder is not None:
            recorder.record(kw["qID"], kw["question"], answers, fmts, preds, rewards)
        return rewards

    return focus_evaluator_reward


# ── model ────────────────────────────────────────────────────────────────────

def load_policy(base_model_id: str, sft_adapter_dir: str | None, init_from_sft: bool):
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.tokenizer.padding_side = "left"  # left-pad for batched generation

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map="auto",
    )

    if init_from_sft:
        if not sft_adapter_dir:
            raise SystemExit("--init-from-sft requires --sft-adapter-dir")
        from peft import PeftModel
        print(f"Warm start: merging SFT adapter {sft_adapter_dir!r} into base weights...")
        # Merge SFT into the base so it becomes the frozen reference the KL
        # anchors to; the fresh GRPO LoRA (added by GRPOTrainer via peft_config)
        # trains on top and starts ~identical to the SFT policy.
        model = PeftModel.from_pretrained(model, sft_adapter_dir).merge_and_unload()
    else:
        print("Cold start: base Qwen weights (KL reference = base).")

    return processor, model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                     choices=["heico", "lapchole"], help="combined = both (default)")
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-name", default=None)

    # the two levers
    ap.add_argument("--init-from-sft", action=argparse.BooleanOptionalAction, default=True,
                     help="warm-start from --sft-adapter-dir (merged) vs cold-start from base")
    ap.add_argument("--sft-adapter-dir",
                     default=str(ORENA_SFT_DIR / "checkpoints" / "heico-only-9b-r16-direct"),
                     help="SFT LoRA checkpoint to warm-start from (used only with --init-from-sft)")
    ap.add_argument("--use-kl", action=argparse.BooleanOptionalAction, default=True,
                     help="apply the GRPO KL penalty (beta) toward the reference policy")
    ap.add_argument("--kl-beta", type=float, default=0.04)

    # prompt
    ap.add_argument("--condition-procedure", action="store_true",
                     help="prepend 'Procedure type: <name>.' to each question")

    # GRPO knobs
    ap.add_argument("--group-size", type=int, default=10, help="completions per prompt (G)")
    ap.add_argument("--batch-size", type=int, default=20,
                     help="per-device completions/step; must be a multiple of --group-size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (calibration/smoke)")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="max completion length")
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temp")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)

    # judge
    ap.add_argument("--skip-open-formats", action="store_true",
                     help="drop open_ended/multiple_choice/matching -> verifiable-only, no judge/API")
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini",
                     help="model id for the open-ended judge, as the provider names it. OpenRouter "
                          "namespaces them, e.g. 'openai/gpt-4o-mini', 'anthropic/claude-3.5-haiku'. "
                          "For the OpenAI API directly, use a bare id like 'gpt-4o-mini'.")
    ap.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1/chat/completions",
                     help="OpenAI-compatible chat-completions endpoint. Default: OpenRouter. "
                          "For OpenAI directly: https://api.openai.com/v1/chat/completions")
    ap.add_argument("--judge-api-key-var", default="OPENAI_API_KEY",
                     help="env var holding the API key (loaded from .env if present)")
    ap.add_argument("--env-file", type=Path, default=GRPO_DIR.parent / ".env",
                     help="path to a .env file to load before reading the API key")
    ap.add_argument("--judge-workers", type=int, default=10)

    ap.add_argument("--rollout-log-steps", type=int, default=10,
                     help="print+log one group's rollouts (question, GT, completions, rewards) "
                          "every N steps")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="orena-frame-grpo")
    args = ap.parse_args()

    if args.batch_size % args.group_size != 0:
        ap.error(f"--batch-size ({args.batch_size}) must be a multiple of --group-size "
                  f"({args.group_size}) so completions divide evenly into groups.")

    # judge
    judge = None
    if not args.skip_open_formats:
        load_dotenv(args.env_file)
        key = os.environ.get(args.judge_api_key_var)
        if not key:
            ap.error(f"open-ended formats need a judge: set {args.judge_api_key_var} (in "
                      f"{args.env_file} or the environment), or pass --skip-open-formats.")
        judge = APIJudge(api_url=args.judge_base_url, api_key=key, model_name=args.judge_model)
        print(f"Open-ended judge: APIJudge({args.judge_model!r}) via {args.judge_base_url}")

    system_prompt = build_system_prompt(style="direct")
    print("=" * 78 + "\nDIRECT SYSTEM PROMPT\n" + "=" * 78)
    print(system_prompt + "=" * 78, flush=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    init_tag = "sftinit" if args.init_from_sft else "cold"
    kl_tag = f"kl{args.kl_beta}" if args.use_kl else "nokl"
    run_name = args.run_name or f"grpo-9b-{init_tag}-{kl_tag}-{timestamp}"
    args.output_dir = args.output_dir or str(GRPO_DIR / "checkpoints" / run_name)

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    print(f"Building dataset from {args.datasets} (TRAIN, frame track)...")
    dataset = build_dataset(args.datasets, cfg, system_prompt,
                            args.condition_procedure, args.skip_open_formats)
    print(f"  {len(dataset)} training prompts.")

    processor, model = load_policy(args.model_id, args.sft_adapter_dir, args.init_from_sft)

    from peft import LoraConfig
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    wandb.init(project=args.wandb_project, name=run_name, config={
        "model_id": args.model_id, "datasets": args.datasets,
        "init_from_sft": args.init_from_sft, "sft_adapter_dir": args.sft_adapter_dir,
        "use_kl": args.use_kl, "kl_beta": args.kl_beta if args.use_kl else 0.0,
        "group_size": args.group_size, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lr": args.lr, "temperature": args.temperature,
        "condition_procedure": args.condition_procedure,
        "skip_open_formats": args.skip_open_formats, "judge_model": args.judge_model,
        "prompt_style": "direct", "n_prompts": len(dataset),
    })

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        # CRITICAL: without this, TRL renders the generation prompt with Qwen's
        # <think> block left OPEN, so the model REASONS instead of answering --
        # every rollout rambles, hits the token cap unterminated, and scores 0
        # (train/inference mismatch: the direct-SFT model was trained with the
        # empty CLOSED block). enable_thinking=False = answer directly.
        chat_template_kwargs={"enable_thinking": False},
        temperature=args.temperature,
        top_p=1.0,
        beta=args.kl_beta if args.use_kl else 0.0,
        scale_rewards=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,  # vllm not installed; generation via transformers
        remove_unused_columns=False,  # keep REWARD_COLS for the reward fn
        report_to="wandb",
        run_name=run_name,
        seed=args.seed,
    )

    recorder = RolloutRecorder()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_fn(judge, args.judge_workers, recorder)],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config,
        callbacks=[RolloutInspectionCallback(recorder, every=args.rollout_log_steps)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nDone. GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
