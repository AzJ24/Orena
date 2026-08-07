"""Build a realistic 20-question /input batch from the real FOCUS TEST split,
in the exact layout the challenge platform mounts, so the container can be
timed end-to-end against the 120s + 20x5s budget.

Mirrors the platform: request.json (list of Requests), frames/<qID>.png,
FO_definitions.json, batch.json. Questions/frames are real test data, not the
template's 3 placeholder samples.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/ajenane/orena/orena_sft")
from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS
from focus.foreign_objects import FO_DEFINITIONS_FILE

from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402

OUT = Path("/home/ajenane/orena/orena_submission/real_batch/interface_1")
N_PER_DATASET = 10  # 10 heico + 10 lapchole = 20, matching a real batch size


def main():
    cfg = FocusConfig(root_dir=DEFAULT_ROOT_DIR)
    set_config(cfg)

    frames_dir = OUT / "frames"
    if OUT.exists():
        shutil.rmtree(OUT)
    frames_dir.mkdir(parents=True)

    requests = []
    n = 0
    for dataset in ("heico", "lapchole"):
        base_fps = float(DATASET_BASE_FPS[dataset])
        ds = FocusDataset(dataset, DatasetSplit.TEST, Track.FRAME)
        taken = 0
        for i in range(len(ds)):
            if taken >= N_PER_DATASET:
                break
            req, _ref = ds[i]
            p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
            if not p.exists():
                continue
            n += 1
            qid = f"q{n:03d}"
            shutil.copy(p, frames_dir / f"{qid}.png")
            requests.append({
                "qID": qid,
                "videoID": req.videoID,
                "start_time": float(req.start_time),
                "end_time": float(req.end_time),
                "procedure_type": req.procedure_type,
                "question": req.question,
            })
            taken += 1
        print(f"  {dataset}: {taken} questions")

    (OUT / "request.json").write_text(json.dumps(requests, indent=2))
    (OUT / "FO_definitions.json").write_text(json.dumps(FO_DEFINITIONS_FILE.read_text()))
    (OUT / "batch.json").write_text(json.dumps({
        "qIDs": [r["qID"] for r in requests],
        "batch_size": len(requests),
        "layout": {"frames": "frames/<qID>.png"},
    }, indent=2))

    from PIL import Image
    sizes = {Image.open(p).size for p in frames_dir.glob("*.png")}
    print(f"\nbatch of {len(requests)} questions -> {OUT}")
    print(f"frame resolutions present: {sorted(sizes)}")
    print(f"budget for this batch: 120 + {len(requests)}*5 = {120 + len(requests)*5}s")


if __name__ == "__main__":
    main()
