# Counting experiments

`number` is ~33% of the FOCUS frame benchmark and the worst format in every run
so far. Four interventions on the 9B SFT setup produced nulls on it: LoRA rank
(r8 vs r16), adapter placement (+linear_attn, attention coverage 8/32 -> 32/32),
visual features (+vision LoRA, which made heico counting *worse*, -0.032 p=.008),
and base model size (9B -> 27B, -0.005/-0.015, both ns).

The error structure points at perception rather than the count-emitting step:
76% of lapchole errors are within +/-1 with a systematic **undercount** bias
(-1: 23.8% vs +1: 16.9%), and the same undercounting shows up in multi-label
`fo_class` (94/333 questions list too few objects vs 19 too many). Instance
counting sits at the majority-class prior ("how many Clips": 0.288 vs 0.283).

## Stage 1 -- does enumerating first change anything? (prompt only, no training)

Force the model to list each instance before committing to a total, so it has to
attend to instances separately instead of emitting one holistic guess.

```
sbatch --export=ALL,PROMPT=enumerate_v1 orena_count/eval_prompt.slurm
sbatch --export=ALL,PROMPT=control_v1   orena_count/eval_prompt.slurm   # only if needed
```

`prompts/enumerate_v1.txt` and `prompts/control_v1.txt` share `_shared_head.txt`
verbatim and differ **only** in the enumeration block, so the comparison isolates
enumeration rather than the `ANSWER:` output format.

### The bar

Base 9B `number` accuracy from `orena_sft/eval_base/`:

| prompt | lapchole | heico |
|---|---|---|
| `direct` | 0.092 | 0.193 |
| `structured` (free reasoning line) | 0.120 | -- |
| `thinking` (free CoT) | 0.097 | -- |
| **best GEPA-evolved** | **0.194** | **0.318** |

**Compare against GEPA, not `direct`.** GEPA is the real prompt-engineering
baseline; beating `direct` only shows the prompt is not broken.

### Decision gate

- Beats GEPA on `number` by a clear margin -> proceed to Stage 2.
- Lands between `direct` and GEPA -> run `control_v1` to separate the
  enumeration effect from the `ANSWER:` format change before deciding.
- At or below GEPA -> stop. Generic reasoning already showed almost nothing
  (`thinking` +0.005), and this would say explicit enumeration adds nothing
  either.

Also watch the other formats: `structured` raised `number` (+0.028) while
*lowering* overall accuracy (0.199 vs 0.233), so a counting gain that costs
`fo_class` is not a win. The prompt asks for a list only on "how many"
questions for this reason.

### Stage 1 RESULT -- failed the gate

| `number` | `direct` | `enumerate_v1` | GEPA (the bar) |
|---|---|---|---|
| lapchole | 0.092 | 0.0957 (+0.004) | **0.194** |
| heico | 0.193 | 0.220 (+0.027) | **0.318** |

Overall accuracy: lapchole 0.211 vs 0.233 (fell), heico 0.235 vs 0.230 (flat).
heico gained a little on counting at no overall cost; lapchole gained nothing and
lost `fo_class` (0.178 vs 0.200). Both are far short of GEPA.

The diagnostic matters more than the score. The model followed the instruction on
55.3% of counting questions, and when it enumerated, **list length matched the
emitted count 400/420 = 95%**. It counts its own list correctly; the list is just
far too short (`gt=5` -> `1. Clip: center` -> `ANSWER: 1`). Forcing enumeration
made the failure legible without fixing it: the loss is upstream, in perception.

## Stage 2 -- teacher ceiling (gate)

    .venv/bin/python orena_count/teacher_ceiling.py --n-per-count 40 \
        --prompt orena_count/prompts/enumerate_defs_v1.txt

### Stage 2 RESULT -- failed the gate, line of work closed

`anthropic/claude-opus-5`, 240 frames stratified by ground-truth count:

| prompt | exact match | undercounts | mean listed at gt=6 |
|---|---|---|---|
| class names (`enumerate_v1`) | 0.089 | 88.6% | 2.35 |
| + definitions (`enumerate_defs_v1`) | 0.114 | 86.9% | 1.92 |

Accuracy decays monotonically with the true count: 0.23-0.28 at gt=1, **0.000 at
gt=6**. A frontier model holding the annotation rulebook lists two objects when
there are six. There is no teacher signal, so rejection-sampled traces cannot be
built and **Stages 3-4 are dead**.

For scale: the fine-tuned 9B gets `number` 0.349 (lapchole) / 0.507 (heico) --
3-4x the frontier model. Counting here is not a reasoning or prompting deficit
that a stronger model resolves; the SFT model has learned something from the
training data that general capability does not supply.

Video context does not obviously help either: SFT 27B scores 0.480 on the segment
track (n=121) vs 0.474 on the frame track (n=1326).

## Stage 2b -- prompt calibration against the measured bias (gate)

The GEPA prompts tell the model instance counts are "frequently OVER-estimated"
and to prefer the lower number -- backwards for this checkpoint, which undercounts
(-1: 23.8% vs +1: 16.9% on lapchole). `direct_countup.txt` is the trained `direct`
prompt plus one calibration block pushing the other way; the diff is that block
alone.

    sbatch --export=ALL,PROMPT=direct_countup,\
      MODEL=<ckpt>,BASE=Qwen/Qwen3.5-9B,MAXTOK=32 orena_count/eval_prompt.slurm

### Stage 2b RESULT -- bias corrected, accuracy unchanged

| lapchole pred-gt | baseline | +countup |
|---|---|---|
| -1 | 23.8% | 18.1% |
| **0** | **35.0%** | **34.2%** |
| +1 | 16.9% | 20.7% |

Mean predicted count 2.90 -> 3.30 against mean gt 3.28 (heico: 2.02 -> 2.36 vs
2.35) -- the mean was calibrated almost exactly. Exact match still fell: 0.3503 ->
0.3424 (p=.670) and 0.4887 -> 0.4759 (p=.282). Mean |error| rose 0.99 -> 1.03.

**The residual is variance, not bias.** The undercount mass moved to the overcount
side without accumulating at zero. One sentence fixes the mean; nothing about the
prompt tells the model *which* frame holds four clips instead of three.

This predicts that every remaining output-side intervention fails the same way --
count-balanced resampling and per-example loss normalization both reshape the
predicted-count distribution, and the distribution is already centred correctly.
With the count distribution perfectly calibrated, exact match is still ~34%.

## Stages 3-4 (NOT BUILT -- gated off by Stage 2)

Kept for the record. Stage 3 was rejection-sampled traces verified by the
ground-truth count: the teacher enumerates without seeing the answer and a trace
is kept only when `len(enumeration) == gt_count`. Stage 4 was SFT on those
traces, which would have unblocked the guard at
`orena_sft/sft_train_qwen_frame_ddp.py:307`.

## Stage 3b -- root-cause diagnostics (SFT 9B predictions, full test sets)

Four measurements against `combined-9b-8r-direct/eval_direct`:

1. **The bias is compression toward the train prior, not undercounting.** Signed
   error by gt count (lapchole): +0.39 at gt=1, ~0 at gt=2-3, -0.82 at gt=4,
   -1.45 at gt=7. Train counts have mean 2.71 (73% are 1-3); the predicted
   marginal centres there. Under weak evidence the model regresses to the prior,
   which is why a uniform calibration sentence (Stage 2b) moved the mean without
   fixing exact match.
2. **`number` is heterogeneous.** Class-count questions ("how many classes")
   score 0.61/0.69 -- they are fo_class in disguise. Sponges/drains (large
   objects) score 0.92/0.98. The deficit is concentrated in instance counting of
   clips: 0.287/0.349, with accuracy ~0 at gt>=5 in both datasets.
3. **Counting and identification errors are independent per frame.** On 305
   heico frames carrying both a number and a fo_class question, both-right
   observed = 84 = expected under independence. Presence detection and
   individuation are separate capabilities; this is why vision LoRA lifted
   fo_class while leaving/hurting number.
4. **Some high counts are hard to observe, but NOT most.** (Revised after the
   Stage 5 audit -- the original claim here, "high counts are unobservable, treat
   gt>=6 as ceiling", was generalised from two cherry-picked frames and is too
   strong.) See Stage 5.

Consistency probe: within (video, subtype, gt) groups of >=3 frames, 79-83% of
groups give conflicting counts (mean var 0.5-0.9) -- the per-frame estimate is
noisy exactly as Stage 2b's variance conclusion requires.

## Stage 4 -- resolution probe (inference-time 2x upscale, both arms)

Frames are natively below the processor's resize band, so the model has always
seen them at native resolution (~400-510 visual tokens; a 20px clip is smaller
than one 32px token). `--min-pixels 2073600` on `evaluate_qwen_frame.py` forces
~2x linear upscale (~2000 tokens). Two arms: SFT (`combined-9b-8r-direct`,
primary readout, but trained at native res) and base 9B (control, cannot be
resolution-anchored).

    sbatch --export=ALL,MODEL=orena_sft/checkpoints/combined-9b-8r-direct orena_count/eval_res.slurm
    sbatch orena_count/eval_res.slurm

### Stage 4 RESULT -- RETRACTED, CONFOUNDED BY PROMPT STYLE

The first version of `eval_res.slurm` did not pass `--prompt-style direct`, so
both 2x arms ran with NO system prompt while every native-res reference used the
`direct` prompt. That prompt is 1667 chars carrying the FO definition (graspers,
scissors, trocars are NOT foreign objects). Two variables moved at once, so
neither arm measured resolution.

The signature is unmistakable in the 2x predictions: `fo_class` answers are
`grasper`, `scissors`, `surgical instrument`, `scalpel` -- precisely the classes
the direct prompt excludes -- and base `fo_class` scored exactly 0.000. The
apparent "adapter breaks at 2x" collapse (lapchole overall 0.551 -> 0.262) is
mostly the missing definition, not resolution anchoring.

The base-arm counting "gain" (0.092 -> 0.234, 0.193 -> 0.375) has a competing
explanation of the same sign: without the FO definition the model counts
instruments too, so mean predicted count rises 0.45 -> 1.40 against gt means of
2.4-3.3. That shift alone raises exact match at every gt level tested, which is
what was mistaken for "objects become visible".

`eval_res.slurm` now defaults to `--prompt-style direct` and writes to
`res<MINPX>_<PSTYLE>_...` so the confounded outputs stay distinguishable.

### Stage 4 CORRECTED -- resolution is null (lapchole; heico pending)

With the prompt style matched, the entire effect disappears. Paired McNemar,
n=2252:

| lapchole `number` | native | 2x | delta | p |
|---|---|---|---|---|
| base 9B (was 0.092 -> 0.234) | 0.0859 | 0.0898 | +0.004 | 0.72 |
| SFT (trained AND evaluated at 2x) | 0.3503 | 0.3724 | +0.022 | 0.23 |

Base overall is identical (0.2163 vs 0.2167, p=1.00); no format moves. The 2x
SFT retrain is likewise a null overall (0.5377 -> 0.5413, p=0.73) with no
significant per-format change (fo_class -0.002 p=0.93, open_ended -0.040 p=0.10,
number +0.022 p=0.23).

The counting error structure barely moves: instance subtype 0.287 -> 0.307,
mean predicted count 2.90 -> 3.08 against gt mean 3.28, and the gt>=4 undercount
persists (gt=5: 0.17 -> 0.15). Doubling visual tokens does not make clips
individuable.

**Resolution joins the null list.** The apparent gain was entirely the missing
FO definition inflating counts (mean pred 0.45 -> 1.40) toward the gt mean --
the Stage 2b variance/prior mechanism reappearing in a new disguise, which is
exactly the failure mode Stage 2b warned about.

## Stage 5 -- visual audit of what is actually in the frame (no GPU)

10 clip-count frames, stratified gt=2..6, viewed at 3x upscale.

Result: **the clips are usually visible.** In a gt=5 heico frame the metallic
clips are plainly countable along the tissue edge; several gt=4-6 frames are
similar. This retracts the Stage 3b "annotation ceiling" claim -- there IS
recoverable headroom, so the residual is not mostly label noise.

What the audit does show is *what* the model confuses. Surgical frames are full
of specular highlights from irrigation droplets: small, bright, round. A clip is
small, bright, and curved-rectangular. The discriminator is **shape at small
scale**, and the errors run both ways -- a gt=2 frame with a clip applier and
heavy droplet glare drew pred=5.

Crucially this is NOT a pixel-budget problem, because Stage 4 doubled the pixels
and changed nothing (p=0.23). The gap is instance individuation: pooled patch
features give the LM a holistic "how much clip-ness is present" signal, not a set
of discrete instances to enumerate. Counting needs binding; next-token prediction
over pooled visual features does not supply it.

**Consequence for method choice.** The mechanism-correct fix is instance-level
supervision (detection/point head, count = number of instances). The dataset does
not support it: `frames_overlay/` is only a burnt-in timestamp, and the `.lance`
files carry `frame/h/w` only -- there are no masks or boxes anywhere. Building
that supervision means new annotation, which is a different project.

## What is left for counting

Every intervention tried has been null: LoRA rank, adapter placement, vision
features, 3x model size, enumeration prompting, and teacher distillation. The
undercount bias is universal -- base 9B, SFT 9B, SFT 27B and Opus-5 all list
fewer objects than are annotated.

Untested, but Stage 2b predicts both are nulls -- they reshape the predicted-count
distribution, which is already centred correctly:
- count-balanced resampling
- per-example loss normalization (`number` is 31% of examples but 13% of the
  loss budget)

Worth establishing before spending more: the **annotation ceiling**. Nobody has
measured inter-annotator agreement on these counts. If two annotators disagree on
how many clips are in a cluttered frame, part of the residual is irreducible and
`number` should be treated as a floor to defend, not a gap to close.
