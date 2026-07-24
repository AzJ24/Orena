# FOCUS Frame Track — Results by generalization regime

**The critical distinction:** a run is only measuring **OOD generalization** if the
test procedure was **never in its training data**. That is *not* true for every run:

| Model trained on | heico test (Sigmoid) | lapchole test (cholecystectomy) |
|---|---|---|
| **heico only** | near-OOD (held-out sub-procedure) | **TRUE far-OOD** ✅ |
| **heico + lapchole** (combined SFT, GRPO) | in-distribution | **in-distribution** ❌ not OOD |
| **base** (no SFT) | OOD | OOD |

So **combined SFT's 0.558 on lapchole is an in-distribution number** — it *trained*
on cholecystectomy. It is not comparable to the heico-only runs, whose lapchole
score is genuine far-OOD. The challenge test is a **novel procedure** → the
**TRUE-OOD** table below is the challenge-relevant one.

Metrics: `overall MEAN` / `situs` / `fo_class` from `summary.csv`. situs n=25 on
lapchole (tiny — wide CIs). Prompt `direct` unless noted. `—` = split not run.

---

## A. TRUE OOD — test procedure NOT in training (the challenge-relevant leaderboard)

Model never SFT'd on lapchole; lapchole = genuine far-OOD. **This tops out ~0.47.**

| # | Run (heico-only SFT + inference-time lever, or base) | OOD ovr | OOD situs | OOD fo | targets |
|---|---|---|---|---|---|
| 1 | heico-only SFT + **FO-priors** inject | **0.468** | 0.262 | **0.489** | fo_class |
| 2 | heico-only SFT + **FO-defs** | 0.455 | 0.071 | **0.494** | fo_class |
| 3 | heico-only SFT + anatomy+FO (both) | 0.444 | 0.524 | 0.468 | both |
| 4 | **heico-only SFT r8 (baseline)** | 0.439 | 0.071 | 0.421 | — |
| 5 | heico-only SFT + RAG anatomy | 0.436 | 0.393 | 0.430 | situs |
| 6 | heico-only SFT + conditioning | 0.434 | 0.429 | 0.421 | situs |
| 7 | heico-only SFT + static anatomy | 0.430 | **0.571** | 0.417 | situs |
| 8 | heico-only SFT r16 | 0.428 | 0.000 | 0.412 | — |
| 9 | heico-only SFT 27B r16 | 0.421 | 0.024 | 0.378 | scale |
| 10 | heico-only SFT r32 | 0.420 | 0.012 | 0.404 | rank |
| 11 | heico-only SFT r16 (saved) / lora1632 | 0.410 | 0.000 | 0.386 | — |
| 12 | heico-only SFT r8 (plain prompt) | 0.405 | 0.179 | 0.365 | prompt |
| 13 | Gemma-4-31B heico-only SFT | 0.394 | 0.226 | 0.348 | model |
| — | *base models (no SFT)* | 0.06–0.25 | 0.46–0.86 | 0.00–0.20 | — |

**Base-model detail (all OOD, format-limited):**

| base run | OOD ovr | OOD situs | OOD fo |
|---|---|---|---|
| Gemma-31B direct | 0.247 | 0.643 | 0.229 |
| Qwen-9B direct | 0.233 | 0.464 | 0.200 |
| Qwen-9B direct + conditioning | 0.232 | **0.738** | 0.200 |
| Qwen-27B direct | 0.228 | 0.738 | 0.186 |
| Qwen-9B + RAG + FO-defs | 0.219 | 0.690 | 0.174 |
| Qwen-9B + FO-defs | 0.219 | 0.595 | 0.181 |
| … plain/structured/thinking variants | 0.11–0.20 | 0.37–0.86 | 0.00–0.15 |
| Qwen-9B plain + conditioning | 0.059 | **0.857** | 0.000 |

**True-OOD ceiling ≈ 0.47.** Best lever is **FO-priors/FO-defs** (fo_class → ~0.49,
41% of questions). situs levers (anatomy/RAG/conditioning) recover situs 0.07 → 0.39–0.57
but move overall little (situs is ~1%). Scale and rank barely matter.

---

## B. IN-DISTRIBUTION — test procedure WAS in training (NOT generalization)

Model trained on lapchole. These scores are **the ceiling if you have the data**,
not OOD performance. Shown for reference only — do not compare to Table A.

| Run | trained on lapchole? | ovr | situs | fo | note |
|---|---|---|---|---|---|
| Combined SFT + FO-defs | ✅ SFT | 0.561 | 0.821 | 0.655 | in-dist |
| **Combined SFT (r8)** | ✅ SFT | 0.558 | **0.821** | 0.643 | in-dist |
| GRPO warm ckpt-400 | ✅ RL only | 0.445 | 0.071 | 0.447 | see note |
| GRPO warm ckpt-200 | ✅ RL only | 0.443 | 0.000 | 0.441 | see note |
| GRPO cold ckpt-400 | ✅ RL only | 0.333 | 0.488 | 0.337 | see note |

**Note on GRPO:** it *saw* lapchole in training, but via RL rewards, not SFT — so it
**did not learn the anatomy** (situs still 0.07/0.00, like true-OOD). Confirms
**knowledge is learned by SFT, not by RL.** Its lapchole is only *nominally*
in-distribution. Combined SFT learned it properly (situs 0.82).

---

## The generalization gap, correctly framed

| | heico (its ID/near-OOD) | lapchole | is lapchole OOD? | real gap |
|---|---|---|---|---|
| heico-only SFT r8 | 0.613 | 0.439 | **yes (far-OOD)** | **−0.17 (real)** |
| Combined SFT | 0.621 | 0.558 | no (in-dist) | −0.06 (not a gap) |

Combined SFT's small "gap" is **not generalization** — it's two in-distribution
scores. The only real generalization measurement here is heico-only → lapchole:
**−0.17**, concentrated in situs (0.76 → 0.07).

---

## The solution (for a genuinely novel test procedure)

Combined SFT's 0.56 is unreachable on an unseen procedure. The solution has three
parts, each measured on the *true*-OOD case:

1. **Maximum procedure diversity in SFT** — not to game the leaderboard, but because
   every added procedure (a) shrinks the distance from a novel test procedure to
   something seen, and (b) teaches the model to condition on procedure instead of
   collapsing to one prior. heico-only (1 proc) → OOD 0.44; more procedures should
   raise the floor on a *novel* one (the diversity-buys-invariance hypothesis).
2. **Inference-time levers** — the only things that help a truly unseen procedure,
   because they need no training on it (keyed on `procedure_type` + the provided
   `FO_definitions.json`). Measured on true-OOD: **FO-priors/defs → fo_class +7pt**;
   **conditioning/RAG → situs 0.07 → 0.4–0.57**. All cheap (RAG retrieval ~0.3 ms).
3. **Leave-one-procedure-out validation** — never validate on a trained procedure.
   heico-only → lapchole IS this protocol; combined → lapchole is NOT.

**Realistic expectation on a novel procedure:**
```
~0.44  diverse-SFT competence (true OOD)
 +0.03 FO-priors (fo_class lift, 41% of questions)
 +situs recovery from conditioning/RAG (small overall weight)
 ≈ ~0.47–0.48   — NOT 0.56
```

## Findings

1. **"Data diversity wins" is real but must be stated correctly:** adding a
   procedure fixes *that* procedure (in-distribution). For a *novel* one, only
   diversity-driven invariance + inference-time levers help — and those cap ~0.47.
2. **Knowledge is learned by SFT, not RL** (GRPO trained on lapchole, situs still 0;
   combined SFT, situs 0.82).
3. **Knowledge/format split:** base knows anatomy (situs ≤0.86) but can't emit
   format (overall <0.25); injections can't rescue a format-non-compliant base.
4. **fo_class injections are the best true-OOD lever** (+7pt, 41% of questions);
   situs levers help situs but barely move overall.
5. **Scale buys nothing; low LoRA rank forgets marginally less.**
6. **`number` unsolved everywhere** (~0.32–0.51) — perception/counting limit.

## Bottom line

For the challenge (novel procedure), the target is **~0.47–0.48**, reached by:
**broadest-possible procedure diversity in SFT + inference-time conditioning/RAG/FO-injection
+ a counting fix**, validated **leave-one-procedure-out**. Combined SFT's 0.56 is the
in-distribution ceiling — achievable only for procedures whose data you have.
