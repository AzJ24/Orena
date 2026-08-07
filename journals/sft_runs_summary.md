# FOCUS frame track — three SFT runs

Same recipe, different training data. The series isolates what training-data
coverage buys.

## Method (identical in all three)

- **Model:** `Qwen/Qwen3.5-9B` (VLM), bf16.
- **Adaptation:** LoRA `r=8, alpha=16, dropout=0.05` on `q/k/v/o_proj` +
  `gate/up/down_proj` — **language model only, vision encoder frozen**. Merged into
  the base weights for eval/submission.
- **Optimization:** lr `1e-4`, **effective batch 32** in every run, loss masked to
  the answer tokens only, best checkpoint by `eval_loss` on a held-out video split.
- **Prompt (`direct`, identical at train and eval):** one frame + one question, the
  FO class registry read live from `FOType.names()`, "answer only, single short
  line", and format rules keyed off the **question wording** — not off a format
  field, which doesn't exist at inference — mirroring the `focus.data.formats`
  parsers. Nothing procedure- or anatomy-specific → OOD-safe.
- **Evaluation:** FOCUS library `Evaluator` — deterministic parsing for
  `binary`/`number`/`fo_class`/`time`, LLM-judge for `open_ended`/`multiple_choice`,
  video-level bootstrapped CIs. Greedy, `max_new_tokens=32`.
- **Why SFT:** base VLMs fail on *format*, not perception (`fo_class` ≈ 0 — prose
  answers the parser rejects). SFT is worth ~+25 points, mostly format compliance.

## Data — and which split is ID vs OOD

HF `orena-dkfz/{heico,lapchole}-focus-vqa`, frame track only. Images referenced by
path; eval carved out of train at the **video** level (question-level would leak
correlated frames).

| export | train | eval | test | procedures |
|---|---|---|---|---|
| `heico` | 7,200 | 800 | 4,000 | train/eval: Proctocolectomy + Rectal Resection · **test: Sigmoid Resection** |
| `lapchole` | 5,023 | 725 | 2,252 | Laparoscopic Cholecystectomy |
| `combined` | 12,225 | 1,523 | 6,252 | union |
| `combined_all` | **20,000 = all splits of both datasets (train+eval+test)** | — | — | union |

heico = colorectal / lower abdomen, lapchole = gallbladder / upper abdomen. The
dataset's own `ood` flag is False everywhere, so **heico→lapchole is the OOD proxy**.

| trained on | heico test | lapchole test |
|---|---|---|
| heico only | near-OOD (unseen sub-procedure) | **TRUE far-OOD** ✅ |
| heico + lapchole | in-distribution | in-distribution ❌ |
| all splits (`combined_all`) | **contaminated** (trained on test) | **contaminated** |

## The runs

| | `heico-only-9b-8r-direct` | `combined-9b-8r-direct` | `combined-all-9b-8r-direct-ddp` |
|---|---|---|---|
| train data | `heico/train` (7.2k) | `combined/train` (12.2k) | `combined_all` (20k, all splits) |
| hardware | 1× RTX PRO 6000 | 1× RTX PRO 6000 | 2× H200, torchrun DDP (16/GPU) |
| best ckpt | 450 (eval_loss 0.339) | 550 (0.326) | 938 = 1.5 ep (0.241, contaminated) |
| purpose | forgetting / true-OOD baseline | data-diversity experiment | submission model |

## Results (`overall MEAN`)

| run | heico test | lapchole test |
|---|---|---|
| heico-only | 0.613 | **0.439** *(true OOD)* |
| combined | 0.621 | 0.558 *(in-distribution)* |
| combined-all-ddp | 0.772 | 0.641 |

> ⚠️ `combined-all-ddp` trained on both test splits — its scores are **not a valid
> measurement**, only a confirmation that the run converged. Its honest expected
> performance is bounded by the combined run.

- **Adding a procedure is the big lever:** lapchole 0.439 → 0.558 (+12 pts), driven
  by recovered cross-procedure anatomy knowledge that heico-only SFT had wiped out.
- **In-distribution is unaffected** (0.613 → 0.621) — the diversity is free.
- **The only real generalization gap is heico-only: 0.613 → 0.439 (−0.17).**
  Combined's −0.06 is two in-distribution scores, not a gap.
- For the challenge's **novel** test procedure the relevant figure is ~0.44, not
  0.56; inference-time levers (FO-priors, procedure conditioning, RAG anatomy) lift
  it to ~0.47–0.48.
