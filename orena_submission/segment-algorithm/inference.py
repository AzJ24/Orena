"""
ORena SAVE FOCUS submission -- SEGMENT track.

Model: Qwen3.6-27B, LoRA-SFT-tuned (r=8) then merged --
`segment-27b-alldata-n80-650-20260730-merged`, trained with the `direct`-style
segment system prompt (`resources/prompts.py`, i.e.
`build_system_prompt(include_definitions=False, style="direct", track="segment")`).

Each question is answered from 80 frames spanning its `[start_time, end_time]`
window at 640x360, with the video metadata that makes Qwen3-VL render ABSOLUTE
source-video timestamps -- the same input `segment_track/collate.py` built during
training. `resources/clip_frames.py` rebuilds that from the trimmed 5 fps clip the
platform provides; see its docstring for why the timeline has to be restored rather
than read from the clip.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# decord must be imported AFTER torch: importing it first breaks CUDA
# initialisation (a known decord issue).
import decord
from focus import Request, Response, load_requests, save_items
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
from resources.clip_frames import (  # noqa: E402
    FRAME_SIZE, N_FRAMES, TIMELINE_FPS, build_metadata, clip_inputs,
)
from resources.prompts import build_system_prompt, extract_answer  # noqa: E402

# The platform always mounts /input and /output; the env overrides exist only so
# this exact file can be dry-run outside a container (test_native.slurm) against
# the same sample batch, rather than testing a second copy of the logic.
INPUT_PATH = Path(os.environ.get("ORENA_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("ORENA_OUTPUT_PATH", "/output"))
MODEL_PATH = Path(os.environ.get("ORENA_MODEL_PATH", RESOURCES_PATH / "segment_model"))

# Both variants show the same footage; training used clean source frames, so the
# plain clip is the one that matches. The overlay's burned-in clock would be an
# unseen artefact -- the timestamps reach the model through the metadata instead.
VIDEO_DIR = INPUT_PATH / "plain"

MAX_NEW_TOKENS = 32  # matches --prompt-style direct at eval time

# The platform allows 120 s setup + 15 s per question, pooled over the batch.
# Overrunning does not fail the run: a growing share of questions is forfeited, and
# only past +20% is the whole batch lost. So the cliff to avoid is 1.2x -- NOT the
# nominal budget. Stopping at the nominal budget would be actively harmful: a
# skipped question scores zero for certain, while an answered one can still be
# right, and stopping cannot refund time already spent. This matters because model
# load alone can exceed the 120 s setup allowance on a small batch.
SETUP_BUDGET_S = 120.0
PER_QUESTION_BUDGET_S = 15.0
FORFEIT_CLIFF = 1.2  # fraction of the budget at which the whole batch is lost
# Assumed cost of a question before any has been timed; offline mean is 2.9 s.
INITIAL_QUESTION_ESTIMATE_S = 6.0

SYSTEM_PROMPT = build_system_prompt(include_definitions=False, style="direct", track="segment")


def clip_path_for(req: Request) -> Path:
    """The clip belonging to this question, already cut to its window."""
    return VIDEO_DIR / f"{req.qID}.mp4"


def answer_one(processor, model, req: Request) -> str:
    """One clip + question -> one parsed answer."""
    reader = decord.VideoReader(str(clip_path_for(req)), ctx=decord.cpu(0), num_threads=1)
    try:
        video, meta = clip_inputs(reader, req.start_time, req.end_time, FRAME_SIZE, N_FRAMES)
    finally:
        del reader

    prompt_text = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": req.question},
            ]},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # do_sample_frames=False: the frames are already sampled, and letting the video
    # processor re-sample would overwrite frames_indices and destroy the timeline.
    inputs = processor(text=[prompt_text], videos=[video], video_metadata=[meta],
                       do_sample_frames=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_answer(raw)


def write_answers(responses: list[Response]) -> None:
    """Write /output/answer.json. Called twice: once with empty placeholders before
    anything risky, then again with the real answers."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / "answer.json"
    save_items(responses, output_path)
    log.info("Wrote %d response(s) to %s", len(responses), output_path)


def warm_up(processor, model) -> None:
    """One throwaway generation on a synthetic clip.

    fla's Gated DeltaNet kernels are Triton, so they JIT-compile on first use. Left
    to happen inside the loop, that cost lands entirely on question 1. The batch
    budget is pooled, so this does not buy time -- it just keeps the compile out of
    a measured question and surfaces it in the log if it is ever slow.
    """
    import numpy as np

    video = np.zeros((N_FRAMES, FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    meta = build_metadata([int(round(t * TIMELINE_FPS))
                           for t in np.linspace(0.0, 16.0, N_FRAMES)], FRAME_SIZE)
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": "Is a foreign object visible?"}]},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], videos=[video], video_metadata=[meta],
                       do_sample_frames=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS - SEGMENT inference start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s", torch.cuda.get_device_name(0))
        log.info("  VRAM total: %.1f GiB",
                 torch.cuda.get_device_properties(0).total_memory / 1024**3)
    else:
        log.warning("No CUDA device -- a 27B model on CPU will not meet the latency budget")

    log.info("--- Loading inputs ---")
    for name in ("batch.json", "request.json", "FO_definitions.json"):
        p = INPUT_PATH / name
        log.info("  %-24s %s", name, f"{p.stat().st_size} bytes" if p.exists() else "MISSING")
    log.info("  %-24s %s", "plain/",
             f"{len(list(VIDEO_DIR.glob('*.mp4')))} clip(s)" if VIDEO_DIR.is_dir() else "MISSING")

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    log.info("Batch of %d question(s)", len(requests))

    # Write a complete-but-empty answer.json NOW, before anything that can fail.
    # Everything below -- loading 51 GB of weights, the first CUDA kernel launch --
    # can die in ways that never reach the writer at the end of this function, and a
    # run that produces no output file at all is a harder failure than one that
    # produces zeros. The real answers overwrite this.
    write_answers([Response(qID=r.qID, content="", latency=0.0) for r in requests])

    # FO_definitions.json is read for the log only. This checkpoint was trained
    # against the static class registry baked into orena-focus (see
    # resources/prompts.py), so the system prompt is fixed; swapping in a
    # different class list here would prompt the model in a way it never saw.
    # Guarded: a shape change here must not cost the batch, since nothing downstream
    # depends on it.
    fo_path = INPUT_PATH / "FO_definitions.json"
    try:
        if fo_path.exists():
            log.info("FO definitions present (%d chars); prompt uses the trained "
                     "static registry", len(json.loads(fo_path.read_text())))
    except Exception:
        log.exception("Could not parse FO_definitions.json; continuing (it is unused)")

    log.info("--- Loading model (once for the batch) ---")
    if not MODEL_PATH.exists():
        log.error("Model not found: %s", MODEL_PATH)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # device_map streams the shards straight onto the GPU; loading to CPU first
    # would need ~55 GB of host RAM before the copy.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": 0} if device.type == "cuda" else None,
    )
    model.eval()
    log.info("Model loaded and set to eval mode (%.2f s)", time.monotonic() - t_start)

    t_warm = time.monotonic()
    try:
        warm_up(processor, model)
        log.info("Warm-up generation done (%.2f s, Triton kernels compiled)",
                 time.monotonic() - t_warm)
    except Exception:
        # Never fatal: a failed warm-up costs the first question some latency,
        # not the batch.
        log.exception("Warm-up failed; continuing without it")

    setup_s = time.monotonic() - t_start
    log.info("Setup complete in %.2f s", setup_s)
    if device.type == "cuda":
        log.info("  VRAM after model load: %.1f GiB allocated",
                 torch.cuda.memory_allocated(0) / 1024**3)

    budget_s = SETUP_BUDGET_S + PER_QUESTION_BUDGET_S * len(requests)
    hard_deadline = t_start + budget_s * FORFEIT_CLIFF
    log.info("Latency budget: %.0f s (%.0f s spent on setup); whole batch forfeited "
             "past %.0f s", budget_s, setup_s, budget_s * FORFEIT_CLIFF)

    log.info("--- Running inference over %d question(s) ---", len(requests))
    responses = []
    n_failed = n_skipped = 0
    spent, answered = 0.0, 0

    for i, req in enumerate(requests, start=1):
        log.info("[%d/%d] qID=%s videoID=%s window=[%.2f, %.2f] s (%.2f s)",
                 i, len(requests), req.qID, req.videoID,
                 req.start_time, req.end_time, req.duration)
        t0 = time.monotonic()

        estimate = spent / answered if answered else INITIAL_QUESTION_ESTIMATE_S
        # Skipping is only ever worth it to STOP the batch crossing the cliff. Once
        # the cliff is already behind us the batch is forfeited whatever we do, so
        # skipping buys nothing and guarantees a zero, while answering at least
        # leaves a scoreable response. Guarding on `t0 <= hard_deadline` is what
        # keeps a slow setup -- which no amount of skipping can refund -- from
        # blanking every question in the batch.
        if t0 <= hard_deadline and t0 + estimate > hard_deadline:
            n_skipped += 1
            log.warning("[%d/%d] qID=%s skipped -- %.0f s elapsed, ~%.1f s needed, "
                        "cliff at %.0f s", i, len(requests), req.qID,
                        t0 - t_start, estimate, budget_s * FORFEIT_CLIFF)
            responses.append(Response(qID=req.qID, content="", latency=0.0))
            continue
        if t0 > hard_deadline:
            log.warning("[%d/%d] qID=%s past the %.0f s cliff already (%.0f s elapsed) "
                        "-- answering anyway, skipping cannot refund it",
                        i, len(requests), req.qID, budget_s * FORFEIT_CLIFF, t0 - t_start)

        try:
            answer = answer_one(processor, model, req)
        except Exception:
            # One bad question must not cost the batch; an empty answer just
            # scores incorrect.
            n_failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting empty answer", i, len(requests), req.qID)
            answer = ""

        latency = time.monotonic() - t0
        spent += latency
        answered += 1
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s answered in %.3f s: %r", i, len(requests), req.qID, latency, answer)

    log.info("Inference complete: %d response(s) (%d failed, %d skipped for budget)",
             len(responses), n_failed, n_skipped)
    if answered:
        log.info("Mean %.2f s/question over %d answered", spent / answered, answered)
    if device.type == "cuda":
        log.info("Peak VRAM: %.1f GiB (platform RTX PRO 6000 Blackwell has 96 GB)",
                 torch.cuda.max_memory_allocated(0) / 1024**3)

    write_answers(responses)
    log.info("=== inference done in %.2f s total ===", time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
