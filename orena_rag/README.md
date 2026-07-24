# orena_rag — procedure-knowledge injection experiments

Isolated experiments testing whether injecting **procedure-specific knowledge**
into the prompt at inference (keyed on `procedure_type`, which is given) recovers
OOD capabilities that SFT forgot — without training data for the target procedure.

Two payloads, each targeting a different capability:
- **anatomy** → `spatial_localization_situs` (+ some spatial / open-ended)
- **fo_priors** → `fo_class` (attacks the FO-prevalence mismatch)

`procedure_knowledge.json` — curated (static-first) knowledge per procedure. This
stands in for what a dynamic retriever (bundled atlas + FAISS) would produce; if
injection helps here, the retrieval pipeline is worth building.

Injection is driven by `evaluate_qwen_frame.py --inject-knowledge {anatomy,fo_priors,both}
--knowledge-file <this json>`; all outputs are written under `orena_rag/runs/`.

## Baselines (already run, elsewhere)
- heico-only r8 SFT, lapchole, **unconditioned**: situs 0.071, fo_class 0.421, overall 0.439
- heico-only r8 SFT, lapchole, **name-only conditioned**: situs 0.429, fo_class 0.421, overall 0.434
