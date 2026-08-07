# `resources/` — the model lives here

Everything in this folder is **copied into the container image** at build time and
is available at inference.

| File | What it is |
|:--|:--|
| `frame_model/` | the merged Qwen3.5-9B + LoRA-SFT checkpoint (`combined-all-9b-8r-direct-ddp`, checkpoint-938) — a full HF model dir (`config.json`, `model.safetensors`, tokenizer/processor files) |
| `prompts.py` | the exact 'direct' system prompt and answer parser this checkpoint was trained/evaluated with (copied from `orena_sft/prompts.py`) |

`inference.py` loads from here via `AutoProcessor`/`Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH)`
and `from resources.prompts import build_system_prompt, extract_answer`.

## Things to get right

**There is no internet at inference time.** The container runs with networking
disabled, so nothing can be downloaded at runtime — no model hubs, no checkpoints,
no tokenizer files, no fonts. Everything your model touches must be inside the
image. If you use a library that lazily fetches weights on first call (many
`transformers`, `timm`, and `open_clip` entry points do), pre-download during the
Docker build and load from a local path instead.

**Load your model once.** One container run answers a whole batch of questions, so
the checkpoint is loaded once and reused across every question in that batch. Keep
model loading outside the per-question loop in `inference.py` — that is where the
setup allowance in the latency budget comes from.

**Large checkpoints need splitting.** A single image layer cannot exceed **50 GB**,
and each Docker `COPY` is one layer. If your weights are bigger than that, split
them into chunks, copy them with several `COPY` instructions, and reassemble at
runtime — see the comment in the `Dockerfile`.

**Big weights make big images.** The whole image is pulled before your code starts.
That pull is infrastructure overhead and is not charged against your latency
budget, but it does slow every submission cycle, so keep the image only as large as
it needs to be.
