"""
ORena SAVE FOCUS submission -- FRAME track.

Model: Qwen3.5-9B, LoRA-SFT-tuned then merged (combined-all-9b-8r-direct-ddp,
checkpoint-938), trained with the 'direct' system prompt (resources/prompts.py,
a copy of orena_sft/prompts.py's build_system_prompt(style="direct")) -- no
RAG / knowledge injection, no procedure conditioning. Evaluation MUST use the
same prompt the model was trained with, so that prompt is reproduced here
byte-for-byte rather than re-derived.
"""

import logging
import sys
import time
from pathlib import Path

import torch
from focus import Request, Response, load_requests, save_items
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESOURCES_PATH = Path(__file__).parent / "resources"
sys.path.insert(0, str(RESOURCES_PATH.parent))
from resources.prompts import build_system_prompt, extract_answer  # noqa: E402

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = RESOURCES_PATH / "frame_model"
FRAME_DIR = INPUT_PATH / "frames"

MAX_NEW_TOKENS = 32  # matches --prompt-style direct default at eval time

SYSTEM_PROMPT = build_system_prompt()


def frame_path_for(req: Request) -> Path:
    return FRAME_DIR / f"{req.qID}.png"


def answer_one(processor, model, device: torch.device, req: Request) -> str:
    """Generate an answer for a single (frame, question) pair."""
    image = Image.open(frame_path_for(req)).convert("RGB")

    text = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": frame_path_for(req)},
                {"type": "text", "text": req.question},
            ]},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return extract_answer(raw)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS — inference start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s", torch.cuda.get_device_name(0))

    log.info("--- Loading inputs ---")
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    log.info("Batch of %d question(s)", len(requests))

    # FO_definitions.json is informational for participants; this checkpoint was
    # trained against the static class list baked into the orena-focus package
    # (see resources/prompts.py), so the system prompt above is fixed and does
    # not depend on this file's contents.

    log.info("--- Loading model (once for the batch) ---")
    if not MODEL_PATH.exists():
        log.error("Model not found: %s", MODEL_PATH)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    log.info("Model loaded and set to eval mode (setup took %.2f s)", time.monotonic() - t_start)

    log.info("--- Running inference over %d question(s) ---", len(requests))
    responses = []
    n_failed = 0

    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        try:
            answer = answer_one(processor, model, device, req)
        except Exception:
            n_failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting empty answer", i, len(requests), req.qID)
            answer = ""
        latency = time.monotonic() - t0
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s answered in %.3f s: %r", i, len(requests), req.qID, latency, answer)

    log.info("Inference complete: %d answered (%d failed)", len(responses), n_failed)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / "answer.json"
    save_items(responses, output_path)
    log.info("Wrote %d response(s) to %s", len(responses), output_path)
    log.info("=== inference done in %.2f s total ===", time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
