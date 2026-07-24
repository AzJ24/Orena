# RAG anatomy retrieval for the FOCUS surgical-VQA challenge

A self-contained description of the idea, the pipeline, the implementation, the
results, and the open levers. Written to be handed to Claude (or a human) as
context for continuing the work.

---

## 1. The problem it solves

The FOCUS FRAME track asks single-frame questions about surgical foreign objects.
One capability — **`spatial_localization_situs`** ("where is X / which anatomical
region") — is **knowledge-heavy**: it needs the anatomy of the *specific procedure*
in the frame.

Two facts about the challenge:
- **The test procedure is OOD** — a procedure the model was likely not trained on.
  We measure this via heico (colorectal) → lapchole (cholecystectomy) transfer.
- **`procedure_type` is given at inference** (it's a field on every `Request`, e.g.
  `"Laparoscopic Cholecystectomy"`).

What we observed:
- **SFT on one procedure destroys the anatomy of others.** heico-only SFT scores
  situs **0.76 in-distribution → 0.07 OOD** (catastrophic forgetting).
- **The base model still knows the anatomy** — base situs on lapchole is 0.46, and
  0.74 when conditioned on `procedure_type`. So SFT *suppressed* the knowledge, it
  didn't erase the model's latent capacity.
- **Adding the procedure's data fixes it** (combined SFT: situs 0.82) — but that
  only works for procedures whose data you have. For a **genuinely novel**
  procedure, you can't add data.

**The idea:** at inference, retrieve the anatomy of whatever `procedure_type` you're
given from a **bundled offline corpus**, and inject it into the prompt. The model
then answers situs by *reading* the anatomy instead of *recalling* it — like
`multiple_choice` transfers well because the options are in the prompt. This works
for **any** procedure, seen or novel, because the retrieval is keyed on the given
procedure string, not on training data.

## 2. Why RAG (vs alternatives)

| approach | generalizes to novel proc? | offline? | notes |
|---|---|---|---|
| Static curated KB (`procedure_knowledge.json`) | ❌ only the ~4 hand-written procs | ✅ | best per-proc quality but doesn't scale |
| LLM-generate anatomy at inference | ✅ | needs a local generator; slow | risks a slow first-question-per-procedure |
| **RAG (retrieval from bundled corpus)** | ✅ | ✅ | ms latency (cached), scales to any proc |

RAG wins under the two challenge constraints — **offline Docker** (no internet) and
**5 s/question** — because retrieval is milliseconds and cached per procedure.

## 3. The pipeline (two phases)

### Phase A — build the index (once, online) — `build_rag_index.ipynb`
```
Wikipedia categories → article titles → lead text → semantic filter → passages → embeddings → rag_index/
```
1. **Harvest by category, not keyword.** Keyword title-matching was too noisy
   ("colon" matched "colonel"). Instead: **discover root categories from seed
   articles** (procedures + anatomy you know you need) by shared category
   membership, then walk each category subtree (`MAX_DEPTH=2`) collecting article
   titles. Categories = editor-curated relevance.
2. **Fetch lead text** for each title via the MediaWiki API (`exintro=1`,
   `exlimit=20`; note: full-text extracts cap at 1/request, intros allow 20).
3. **Semantic relevance filter** — depth-2 walks drift off-topic; keep only articles
   whose embedding is near "surgical anatomy" anchors (drops ~57% junk). Protect
   seeds.
4. **Chunk** each article into ~120-word passages, **title-prefixed**
   (`"Cholecystectomy: …"`) so retrieved text is self-describing.
5. **Embed** every passage with MiniLM (`all-MiniLM-L6-v2`, 384-dim, unit-normalized).
6. **Save** `rag_index/{passages.jsonl, embeddings.npy, titles.txt, meta.json}`
   (+ optional `index.faiss`). Current corpus: **1,282 articles → 2,013 passages**.

**Coverage verification (built in):** a coverage-test cell checks that must-have
anatomy articles are present; anything missing names the category branch to add.
An end-to-end validation cell checks coverage/correctness/abstention on a gold set.

**Offline note:** the harvest needs internet (build phase). On an internet-less
kernel it is run once on a connected node and cached; the notebook's fetch cell
loads `rag_index/articles.jsonl` if present.

### Phase B — retrieve + inject (per question) — `retriever.py` + eval scripts
For a request `{procedure_type, question, image}`:
1. **`AnatomyRetriever.facts(procedure_type)`**:
   - cache check (retrieval happens **once per procedure**, then cached).
   - **query expansion**: `procedure_type + " surgical anatomy"` (biases away from
     non-surgical senses, e.g. sigmoid *colon* vs *sinus*).
   - embed query → **cosine** vs all passage embeddings (`embeddings @ q`) → top-4.
   - **abstain**: if top score `< MIN_SCORE (0.58)` → try **morphological fallback**
     (strip surgical suffix/prefix: `cholecyst-ectomy → cholecyst`) → if still
     below → return `""`. *Injecting nothing beats injecting wrong anatomy.*
   - else: join top-4 passage texts, truncate to 600 chars, cache, return.
2. **`build_question(...)`** prepends the block:
   ```
   Procedure type: Laparoscopic Cholecystectomy.
   Relevant anatomy: Cholecystectomy: … Gallbladder: … inferior surface of the liver …
   <original question>
   ```
3. Assemble chat (system = direct prompt; user = image + injected question) →
   VLM `generate()` → `extract_answer()` → scored by the FOCUS `Evaluator`.

## 4. Implementation files

| file | role |
|---|---|
| `orena_rag/build_rag_index.ipynb` | Phase A — build/validate the index (run top-to-bottom) |
| `orena_rag/retriever.py` | `AnatomyRetriever` — load index, `facts(procedure_type)` |
| `orena_rag/rag_index/` | corpus + embeddings (bundle this offline) |
| `orena_rag/procedure_knowledge.json` | static curated KB (the `anatomy`/`fo_priors` fallback path) |
| `orena_sft/evaluate_qwen_frame.py` | eval; `--inject-knowledge rag --rag-index <dir>` |
| `orena_sft/base_model_eval.py` | base-model eval; same flags |

**Key parameters** (in `retriever.py` / notebook): `EMBED_MODEL=all-MiniLM-L6-v2`,
`MIN_SCORE=0.58` (calibrated: real matches ~0.6+, wrong/negatives <0.55),
`QUERY_SUFFIX=" surgical anatomy"`, `TOP_K=4`, `max_chars=600`, `MAX_DEPTH=2`,
relevance-filter threshold `0.25`.

**Eval usage:**
```
--prompt-style direct --inject-knowledge rag \
--rag-index /home/ajenane/orena/orena_rag/rag_index
```
`build_question` also supports static modes: `--inject-knowledge {anatomy,fo_priors,both}`
(from `procedure_knowledge.json`) and `--fo-definitions` (full FO class descriptions
in the system prompt). Predictions log `retrieve_time` and `full_time`.

## 5. Results (heico-only SFT, lapchole = TRUE OOD)

| condition | situs | fo_class | overall |
|---|---|---|---|
| unconditioned | 0.071 | 0.421 | 0.439 |
| procedure conditioning (name only) | 0.429 | 0.421 | 0.434 |
| static curated anatomy | **0.571** | 0.417 | 0.430 |
| **RAG-retrieved anatomy** | 0.393 | 0.430 | 0.436 |

- **RAG recovers situs 0.07 → 0.39** — real, but below the hand-tuned static block
  (0.57). At n=25 all three (0.43/0.57/0.39) are within overlapping CIs.
- **Latency**: `retrieve_time` 0.3 ms, `full_time` 214 ms (amortized; cached per
  procedure) — negligible vs the 5 s budget.
- **Validation**: 9/9 procedures retrieve correct anatomy, 3/3 nonsense abstain.
  Sigmoid→sigmoid colon, Whipple→pancreaticoduodenectomy (both fixed by the clean
  corpus; earlier thin/dirty corpus gave brain-sinus / hemorrhoids).

## 6. Known limitation + the main next lever

RAG underperforms the static block because the query retrieves the **procedure
article** (e.g. "Cholecystectomy" — describes the *operation*: "8th most common
procedure, postcholecystectomy syndrome…"), whose lead is **not the anatomical
layout**. The static block was pure situs vocabulary ("gallbladder, cystic duct,
Calot's triangle, liver bed").

**Fixes to try (in priority):**
1. **Bias retrieval toward anatomy, not procedure** — query the *organs/structures*
   (e.g. also retrieve "Gallbladder", "Cystic duct" articles), or use full article
   text (anatomy sections), not just the procedure lead.
2. **Biomedical embedder** (`pritamdeka/S-PubMedBert-MS-MARCO`, `BAAI/bge-small`) —
   sharper medical retrieval; recalibrate `MIN_SCORE` after swapping.
3. **Gate by question type** — inject anatomy only for situs/"where" questions;
   inject `fo_priors` for fo_class questions; nothing for counting (injection dents
   `number`). Route on question text (no format leak).

## 7. Scope: what RAG does and doesn't help

- **Helps**: situs (procedure anatomy), a little spatial/open-ended.
- **Does not help**: `fo_class` (recognition ≠ anatomy — flat under anatomy
  injection; use `fo_priors`/`FO_definitions.json` instead), `number` (perception/
  counting limit), format compliance (needs SFT — RAG can't rescue a base model:
  base+RAG+FO-defs overall stayed 0.219).
- **Weight caveat**: situs is ~1% of questions, so RAG's overall micro impact is
  small unless the challenge metric is capability-macro. **Confirm the official
  aggregation.**

## 8. How to rebuild / extend

- **Rebuild the index**: run `build_rag_index.ipynb` top-to-bottom (harvest cells
  need internet; they auto-skip if `rag_index/articles.jsonl` exists and the fetch
  cell loads the cache). Re-run chunk→embed→index (§2–4) to regenerate
  `passages.jsonl` + `embeddings.npy`. **These must match `articles.jsonl`** — the
  retriever loads them from disk.
- **Add coverage**: add procedures/structures to `SEEDS`; the coverage-test cell
  shows missing must-haves and the category to add. `MAX_DEPTH` trades recall vs
  drift (drift is removed by the semantic filter).
- **Bundle for Docker**: ship `rag_index/{passages.jsonl, embeddings.npy}` + the
  MiniLM model; load once (setup budget); `facts()` is cached per procedure (ms).

## 9. One-line summary

At inference, key on the given `procedure_type`, semantically retrieve that
procedure's anatomy from a bundled offline Wikipedia-derived index, and inject it —
recovering forgotten situs on any procedure (seen or novel) at ~0 latency, with
abstention so it never injects wrong anatomy. Current gain situs 0.07→0.39; the main
lever to close the gap to the static 0.57 is retrieving *anatomy* articles rather
than *procedure* articles.
