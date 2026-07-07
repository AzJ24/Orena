"""Builds a HuggingFace-`datasets`-ready vision-encoder-probing suite from the
Orena FOCUS (heico) VQA data.

Reads the cached QA dataframe produced by `webapp/data.py` (one row per FOCUS
question) plus the pre-extracted frame JPEGs on disk, and derives a set of
clean single-task classification configs:

  Tier 1 — single image:
    fo_class_identification, fo_class_identification_multilabel,
    quadrant_localization, closest_to_center, occlusion_state,
    sponge_saturation, object_color, grasped_by_instrument, object_count

  Tier 2 — image pair:
    instance_reidentification

  Tier 3 — frame sequence:
    interaction_recognition, temporal_ordering_pairwise

Each config gets its own train/test `datasets.DatasetDict`, split at the
*video* level (never split within a video) so adjacent/correlated frames
never leak across the split.

This script only reads from disk and writes to `--out-dir` (via
`save_to_disk`). It does NOT push anything to the Hub. Pushing is wired up
behind `--push` for later use, once a repo has actually been created.

Usage:
    .venv/bin/python scripts/build_probing_dataset.py --out-dir ./probing_export
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value

# ── paths / constants ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "webapp" / "cache" / "qa_cache.parquet"
ROOT_DIR = Path("/projects/datasets_ML/orena/")
DATASET = "heico"
FRAMES_DIR = ROOT_DIR / DATASET / "frames"
BASE_FPS = 25.0

# Foreign-object class vocabulary used throughout the FOCUS questions.
# Sorted longest-first so substring matching picks "Specimen bag" before
# the shorter "Specimen".
FO_CLASSES = sorted(
    ["Sponge", "Clip", "External drain", "Needle", "Silicone loop", "Specimen bag", "Specimen"],
    key=len, reverse=True,
)

TS_RE = re.compile(r"\d{2}:\d{2}:\d{2}")


# ── small parsing helpers ──────────────────────────────────────────────────

def ts_to_seconds(ts: str) -> float:
    # converts a clock timestamp to seconds
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def find_timestamps(text: str) -> list[float]:
    # finds all HH:MM:SS patterns in a question and returns them as seconds. 
    return [ts_to_seconds(t) for t in TS_RE.findall(text)]


def find_target_fo_class(text: str) -> str | None:
    # finds which object name (Sponge, Clip…) the question is about.
    for cls in FO_CLASSES:
        if cls in text:
            return cls
    return None


def frame_path(video_id: str, t: float) -> Path:
    # returns the path to the frame JPEG for a given video and timestamp, rounded to the nearest frame index.
    video_stem = Path(video_id).stem
    frame_idx = round(t * BASE_FPS)
    return FRAMES_DIR / video_stem / f"frame{frame_idx:07d}.jpg"


def sample_frame_paths(video_id: str, t_start: float, t_end: float, n: int) -> list[Path]:
    """Uniformly sample up to n frame paths within [t_start, t_end] that exist on disk."""
    if t_end <= t_start:
        t_end = t_start + 0.5
    times = np.linspace(t_start, t_end, n)
    paths = []
    for t in times:
        p = frame_path(video_id, t)
        if p.exists():
            paths.append(p)
    return paths


def normalize_yes_no(answer: str) -> str | None:
    a = answer.strip().lower()
    if a.startswith("yes"):
        return "yes"
    if a.startswith("no"):
        return "no"
    return None


# ── video-level split ───────────────────────────────────────────────────────

def make_video_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> dict[str, str]:
    """Assign every videoID to train/test, stratified by procedure_type so both
    splits cover all procedure types, and never splitting a video's frames
    across train and test."""
    rng = np.random.RandomState(seed)
    video_to_proc = df.drop_duplicates("videoID").set_index("videoID")["procedure_type"]
    split: dict[str, str] = {}
    for proc, group in video_to_proc.groupby(video_to_proc):
        videos = group.index.tolist()
        rng.shuffle(videos)
        n_test = max(1, round(len(videos) * test_frac))
        for v in videos[:n_test]:
            split[v] = "test"
        for v in videos[n_test:]:
            split[v] = "train"
    return split


# ── generic dataset assembly ────────────────────────────────────────────────

def _make_split(records: list[dict], features: Features) -> Dataset:
    if not records:
        # Dataset.from_list([], features=...) can't infer columns from zero rows
        # and ends up mismatched against the given Features; build explicitly
        # empty typed columns instead so an (unlikely) empty split still works.
        return Dataset.from_dict({col: [] for col in features}, features=features)
    return Dataset.from_list(records, features=features)


def build_split_dict(records: list[dict], features: Features) -> DatasetDict:
    train_records = [{k: v for k, v in r.items() if k != "_split"} for r in records if r["_split"] == "train"]
    test_records = [{k: v for k, v in r.items() if k != "_split"} for r in records if r["_split"] == "test"]
    return DatasetDict({
        "train": _make_split(train_records, features),
        "test": _make_split(test_records, features),
    })


# Tasks with fewer rows than this (after filtering/cleaning) are too sparse to
# be a usable standalone probing config and are skipped with a warning instead
# of silently shipping a near-empty dataset.
MIN_TASK_ROWS = 20


def maybe_build(name: str, recs: list[dict], dropped: int, label_key: str, features: Features) -> DatasetDict | None:
    report(name, recs, label_key, dropped)
    if len(recs) < MIN_TASK_ROWS:
        print(f"    -> SKIPPED: only {len(recs)} usable rows (< {MIN_TASK_ROWS}), too sparse for a standalone config.")
        return None
    return build_split_dict(recs, features)


def report(task_name: str, records: list[dict], label_key: str, dropped: int) -> None:
    n_train = sum(1 for r in records if r["_split"] == "train")
    n_test = sum(1 for r in records if r["_split"] == "test")
    if records and isinstance(records[0].get(label_key), list):
        flat = [lbl for r in records for lbl in r[label_key]]
    else:
        flat = [r.get(label_key) for r in records]
    dist = Counter(flat)
    print(f"\n[{task_name}] train={n_train} test={n_test} dropped={dropped}")
    for k, v in dist.most_common():
        print(f"    {k!r}: {v}")


# ── Tier 1: single image tasks ──────────────────────────────────────────────

def build_fo_class_identification(df: pd.DataFrame, video_split: dict, multilabel: bool) -> tuple[list[dict], int]:
    """Single-frame FO class identification.

    Restricted to OBJECT_IDENTIFICATION questions only (closest_to_center,
    DURATION_ESTIMATION, and MULTI_STEP_REASONING are excluded — those last two
    ask about a whole time window ("longest duration in this video", "co-occur
    throughout the video"), not a single instant, so pairing them with one
    frame at start_time would be a semantically mismatched (image, label) pair.
    See build_fo_class_identification_window / build_longest_duration_class /
    build_co_occurrence_prediction for the window-correct versions of those.

    Within OBJECT_IDENTIFICATION, frame-track rows are single-frame by
    construction. Segment-track rows are only included here if the question
    pins one explicit instant ("... at HH:MM:SS ...") — that timestamp is used
    instead of the row's start_time. Segment-track rows with zero or two+
    timestamps (vaguer "in this video" / "between X and Y" phrasing) go to
    build_fo_class_identification_window instead.
    """
    sub = df[(df["format"] == "fo_class") & (df["primary_capability"] == "OBJECT_IDENTIFICATION")]

    records, dropped = [], 0
    label_names = FO_CLASSES + ["none"]
    for _, r in sub.iterrows():
        labels = [x.strip() for x in str(r["answer"]).split(",")]
        is_multi = len(labels) > 1
        # fo_class_identification (single-label) only wants single-answer rows;
        # fo_class_identification_multilabel wants every row (both single and
        # multi answers are valid label sets there).
        if not multilabel and is_multi:
            dropped += 1
            continue
        if any(lbl not in label_names for lbl in labels):
            dropped += 1
            continue

        if r["track"] == "frame":
            t = r["start_time"]
        else:
            ts = find_timestamps(r["question"])
            if len(ts) != 1:
                continue  # belongs to build_fo_class_identification_window, not counted as dropped
            t = ts[0]

        p = frame_path(r["videoID"], t)
        if not p.exists():
            dropped += 1
            continue
        rec = {
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(t * BASE_FPS), "timestamp_s": float(t),
            "window_duration_s": float(r["duration"]),
            "_split": video_split.get(r["videoID"], "train"),
        }
        if multilabel:
            rec["labels"] = labels
        else:
            rec["label"] = labels[0]
        records.append(rec)
    return records, dropped


def build_fo_class_identification_window(df: pd.DataFrame, video_split: dict, n_frames: int = 6) -> tuple[list[dict], int]:
    """Multi-label FO class identification over a time window.

    Covers the OBJECT_IDENTIFICATION/fo_class segment-track questions that
    build_fo_class_identification excludes because they reference the whole
    question window rather than one pinned instant (e.g. "Which foreign
    object classes appear in this video?", "...between 02:51:24 and
    02:52:25?"). Frames are sampled uniformly across [start_time, end_time].
    """
    sub = df[(df["format"] == "fo_class") & (df["primary_capability"] == "OBJECT_IDENTIFICATION") & (df["track"] == "segment")]

    records, dropped = [], 0
    label_names = FO_CLASSES + ["none"]
    for _, r in sub.iterrows():
        if len(find_timestamps(r["question"])) == 1:
            continue  # handled by build_fo_class_identification instead
        labels = [x.strip() for x in str(r["answer"]).split(",")]
        if any(lbl not in label_names for lbl in labels):
            dropped += 1
            continue
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]), "n_frames": len(paths),
            "labels": labels,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_co_occurrence_prediction(df: pd.DataFrame, video_split: dict, n_frames: int = 6) -> tuple[list[dict], int]:
    """Multi-label: which FO classes co-occur with the last-seen class, sampled
    across the question's time window (MULTI_STEP_REASONING/fo_class)."""
    sub = df[(df["primary_capability"] == "MULTI_STEP_REASONING") & (df["format"] == "fo_class")]

    records, dropped = [], 0
    label_names = FO_CLASSES + ["none"]
    for _, r in sub.iterrows():
        labels = [x.strip() for x in str(r["answer"]).split(",")]
        if any(lbl not in label_names for lbl in labels):
            dropped += 1
            continue
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]), "n_frames": len(paths),
            "labels": labels,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_longest_duration_class(df: pd.DataFrame, video_split: dict, n_frames: int = 6) -> tuple[list[dict], int]:
    """Single-label: which FO class is visible for the longest total duration
    within the question's time window (DURATION_ESTIMATION/fo_class)."""
    sub = df[(df["primary_capability"] == "DURATION_ESTIMATION") & (df["format"] == "fo_class")]

    records, dropped = [], 0
    label_names = FO_CLASSES + ["none"]
    for _, r in sub.iterrows():
        label = str(r["answer"]).strip()
        if label not in label_names:
            dropped += 1
            continue
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]), "n_frames": len(paths),
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


QUADRANTS = ["top/left", "top/right", "bottom/left", "bottom/right"]


def build_quadrant_localization(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    """Single-frame quadrant localization.

    Restricted to duration == 0 rows. The question template "Where is the
    center of the X located ... in this frame?" is reused verbatim for both
    frame-track (duration 0, literally one frame) and segment-track rows —
    but for the segment-track instances the row's window can be up to 299s
    (e.g. "The last time a Sponge is visible, where was..." / "When X first
    appears in this video, ..." / "...for most of the time"). Those genuinely
    require searching/aggregating across the window, not one arbitrary frame,
    so they're excluded here and handled by build_quadrant_localization_window.
    """
    sub = df[(df["primary_capability"] == "SPATIAL_LOCALIZATION_CAMERA") & (df["format"] == "multiple_choice")]
    sub = sub[~sub["question"].str.contains("during the whole video", case=False)]
    sub = sub[sub["duration"] == 0]

    records, dropped = [], 0
    for _, r in sub.iterrows():
        answer = str(r["answer"]).strip()
        if "," in answer or answer not in QUADRANTS:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": answer,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_quadrant_localization_window(df: pd.DataFrame, video_split: dict, n_frames: int = 6) -> tuple[list[dict], int]:
    """Window-level counterpart of quadrant_localization: same single-quadrant
    label, but frames are sampled across [start_time, end_time] since the
    answer requires finding/aggregating a specific moment within that window
    (first appearance, last appearance, or majority position)."""
    sub = df[(df["primary_capability"] == "SPATIAL_LOCALIZATION_CAMERA") & (df["format"] == "multiple_choice")]
    sub = sub[~sub["question"].str.contains("during the whole video", case=False)]
    sub = sub[sub["duration"] > 0]

    records, dropped = [], 0
    for _, r in sub.iterrows():
        answer = str(r["answer"]).strip()
        if "," in answer or answer not in QUADRANTS:
            dropped += 1
            continue
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]), "n_frames": len(paths),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": answer,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_closest_to_center(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "SPATIAL_LOCALIZATION_CAMERA") & (df["format"] == "fo_class")
             & df["question"].str.contains("closest", case=False)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        answer = str(r["answer"]).strip()
        if answer not in FO_CLASSES:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "label": answer,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def _normalize_occlusion(answer: str) -> str | None:
    a = answer.lower()
    if "recogniz" in a:
        return "fully_recognizable"
    if "blood" in a:
        return "heavily_blood_covered"
    if "blur" in a:
        return "heavily_blurred"
    if "out of sight" in a:
        return "mostly_out_of_sight"
    if "obscur" in a:
        return "partially_obscured"
    return None


def build_occlusion_state(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "OBJECT_ATTRIBUTES")
             & df["question"].str.contains("clearly visible are the defining visual features", case=False)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        label = _normalize_occlusion(str(r["answer"]))
        if label is None:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_sponge_saturation(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "OBJECT_ATTRIBUTES")
             & df["question"].str.contains("state of the Sponge", case=False)]
    mapping = {"clean": "clean", "partially bloody": "partially_bloody", "fully saturated": "fully_saturated"}
    records, dropped = [], 0
    for _, r in sub.iterrows():
        a = str(r["answer"]).strip().lower().rstrip(".")
        label = mapping.get(a)
        if label is None:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_object_color(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "OBJECT_ATTRIBUTES")
             & df["question"].str.contains("color|colour", case=False, regex=True)
             & df["question"].str.contains("predominantly", case=False)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        a = str(r["answer"]).strip().lower().rstrip(".")
        if not a or "," in a:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": a.replace(" ", "_"),
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_grasped_by_instrument(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[df["question"].str.contains("grasped by an instrument", case=False)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        label = normalize_yes_no(str(r["answer"]))
        if label is None:
            dropped += 1
            continue
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def _count_type_from_question(question: str) -> str:
    if "instances" in question:
        return "instances"
    if "classes" in question:
        return "classes"
    for cls in FO_CLASSES:
        plural = cls + "s"
        if plural in question or cls in question:
            return cls
    return "unknown"


def build_object_count(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    """Single-frame object count. Restricted to duration == 0 rows — for
    OBJECT_AGGREGATION/number this happens to coincide exactly with track ==
    "frame" (every segment-track row here has duration >= 17s and asks about
    a video-wide total/maximum/cumulative count, e.g. "How many distinct
    Sponges are visible in this video?", "What is the maximum number of Clips
    appearing at once in a single frame?" — neither is answerable from one
    frame). Those go to build_object_count_window instead."""
    sub = df[(df["primary_capability"] == "OBJECT_AGGREGATION") & (df["format"] == "number") & (df["duration"] == 0)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        try:
            val = int(str(r["answer"]).strip())
        except ValueError:
            dropped += 1
            continue
        label = "4+" if val >= 4 else str(val)
        p = frame_path(r["videoID"], r["start_time"])
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"], "image": str(p),
            "frame_index": round(r["start_time"] * BASE_FPS), "timestamp_s": float(r["start_time"]),
            "window_duration_s": float(r["duration"]),
            "count_type": _count_type_from_question(r["question"]),
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_object_count_window(df: pd.DataFrame, video_split: dict, n_frames: int = 6) -> tuple[list[dict], int]:
    """Window-level counterpart of object_count: video-wide/cumulative/peak
    counts, sampled across [start_time, end_time]."""
    sub = df[(df["primary_capability"] == "OBJECT_AGGREGATION") & (df["format"] == "number") & (df["duration"] > 0)]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        try:
            val = int(str(r["answer"]).strip())
        except ValueError:
            dropped += 1
            continue
        label = "4+" if val >= 4 else str(val)
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]), "n_frames": len(paths),
            "count_type": _count_type_from_question(r["question"]),
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


# ── Tier 2: image pair ───────────────────────────────────────────────────────

def build_instance_reidentification(df: pd.DataFrame, video_split: dict) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "INSTANCE_MATCHING") & (df["format"] == "binary")]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        ts = find_timestamps(r["question"])
        if len(ts) != 2:
            dropped += 1
            continue
        label = normalize_yes_no(str(r["answer"]))
        if label is None:
            dropped += 1
            continue
        t1, t2 = ts
        p1, p2 = frame_path(r["videoID"], t1), frame_path(r["videoID"], t2)
        if not (p1.exists() and p2.exists()):
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "image_1": str(p1), "image_2": str(p2),
            "frame_index_1": round(t1 * BASE_FPS), "frame_index_2": round(t2 * BASE_FPS),
            "dt_seconds": float(t2 - t1),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


# ── Tier 3: frame sequence ───────────────────────────────────────────────────

ACTION_NORMALIZE = {
    "retrieved": "retrieved",
    "created": "created",
}


def _action_type_from_question(question: str) -> str:
    m = re.search(r"being (.+?) at that moment", question)
    if not m:
        return "unknown"
    phrase = m.group(1).lower()
    for key in ACTION_NORMALIZE:
        if phrase.startswith(key):
            return ACTION_NORMALIZE[key]
    if phrase.startswith("inserted"):
        return "inserted"
    return phrase.split()[0]


def build_interaction_recognition(df: pd.DataFrame, video_split: dict, n_frames: int = 5, window_s: float = 1.0) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "FO_INTERACTION_RECOGNITION") & (df["format"] == "binary")]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        ts = find_timestamps(r["question"])
        if len(ts) != 1:
            dropped += 1
            continue
        label = normalize_yes_no(str(r["answer"]))
        if label is None:
            dropped += 1
            continue
        event_time = ts[0]
        paths = sample_frame_paths(r["videoID"], event_time - window_s, event_time + window_s, n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(event_time * BASE_FPS - window_s * BASE_FPS) + i for i in range(len(paths))],
            "start_time": float(event_time - window_s), "end_time": float(event_time + window_s),
            "n_frames": len(paths),
            "event_time": float(event_time),
            "action_type": _action_type_from_question(r["question"]),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


def build_temporal_ordering_pairwise(df: pd.DataFrame, video_split: dict, n_frames: int = 8) -> tuple[list[dict], int]:
    sub = df[(df["primary_capability"] == "TEMPORAL_ORDERING") & (df["format"] == "binary")]
    records, dropped = [], 0
    for _, r in sub.iterrows():
        label = normalize_yes_no(str(r["answer"]))
        if label is None:
            dropped += 1
            continue
        paths = sample_frame_paths(r["videoID"], r["start_time"], r["end_time"], n_frames)
        if not paths:
            dropped += 1
            continue
        records.append({
            "qID": r["qID"], "videoID": r["videoID"], "procedure_type": r["procedure_type"],
            "source_track": r["track"], "source_capability": r["primary_capability"],
            "question": r["question"],
            "images": [str(p) for p in paths],
            "frame_indices": [round(t * BASE_FPS) for t in np.linspace(r["start_time"], r["end_time"], len(paths))],
            "start_time": float(r["start_time"]), "end_time": float(r["end_time"]),
            "n_frames": len(paths),
            "target_fo_class": find_target_fo_class(r["question"]) or "any",
            "label": label,
            "_split": video_split.get(r["videoID"], "train"),
        })
    return records, dropped


# ── feature schemas ──────────────────────────────────────────────────────────

def tier1_features(label_names: list[str], extra: dict | None = None) -> Features:
    base = {
        "qID": Value("string"), "videoID": Value("string"), "procedure_type": Value("string"),
        "source_track": Value("string"), "source_capability": Value("string"), "question": Value("string"),
        "image": Image(), "frame_index": Value("int32"), "timestamp_s": Value("float32"),
        "label": ClassLabel(names=label_names),
    }
    if extra:
        base.update(extra)
    return Features(base)


def tier1_multilabel_features(label_names: list[str]) -> Features:
    return Features({
        "qID": Value("string"), "videoID": Value("string"), "procedure_type": Value("string"),
        "source_track": Value("string"), "source_capability": Value("string"), "question": Value("string"),
        "image": Image(), "frame_index": Value("int32"), "timestamp_s": Value("float32"),
        "window_duration_s": Value("float32"),
        "labels": Sequence(ClassLabel(names=label_names)),
    })


def tier2_features(label_names: list[str]) -> Features:
    return Features({
        "qID": Value("string"), "videoID": Value("string"), "procedure_type": Value("string"),
        "source_track": Value("string"), "source_capability": Value("string"), "question": Value("string"),
        "image_1": Image(), "image_2": Image(),
        "frame_index_1": Value("int32"), "frame_index_2": Value("int32"), "dt_seconds": Value("float32"),
        "target_fo_class": Value("string"),
        "label": ClassLabel(names=label_names),
    })


def tier3_features(label_names: list[str], extra: dict | None = None, with_target_fo_class: bool = True) -> Features:
    base = {
        "qID": Value("string"), "videoID": Value("string"), "procedure_type": Value("string"),
        "source_track": Value("string"), "source_capability": Value("string"), "question": Value("string"),
        "images": Sequence(Image()), "frame_indices": Sequence(Value("int32")),
        "start_time": Value("float32"), "end_time": Value("float32"), "n_frames": Value("int32"),
        "label": ClassLabel(names=label_names),
    }
    if with_target_fo_class:
        base["target_fo_class"] = Value("string")
    if extra:
        base.update(extra)
    return Features(base)


def tier3_multilabel_features(label_names: list[str], extra: dict | None = None) -> Features:
    base = {
        "qID": Value("string"), "videoID": Value("string"), "procedure_type": Value("string"),
        "source_track": Value("string"), "source_capability": Value("string"), "question": Value("string"),
        "images": Sequence(Image()), "frame_indices": Sequence(Value("int32")),
        "start_time": Value("float32"), "end_time": Value("float32"), "n_frames": Value("int32"),
        "labels": Sequence(ClassLabel(names=label_names)),
    }
    if extra:
        base.update(extra)
    return Features(base)


# ── orchestration ────────────────────────────────────────────────────────────

def build_all(df: pd.DataFrame, video_split: dict) -> dict[str, DatasetDict]:
    out: dict[str, DatasetDict] = {}

    # Tier 1 — single image
    recs, dropped = build_fo_class_identification(df, video_split, multilabel=False)
    out["fo_class_identification"] = maybe_build(
        "fo_class_identification", recs, dropped, "label",
        tier1_features(FO_CLASSES + ["none"], {"window_duration_s": Value("float32")}),
    )

    recs, dropped = build_fo_class_identification(df, video_split, multilabel=True)
    out["fo_class_identification_multilabel"] = maybe_build(
        "fo_class_identification_multilabel", recs, dropped, "labels",
        tier1_multilabel_features(FO_CLASSES + ["none"]),
    )

    recs, dropped = build_quadrant_localization(df, video_split)
    out["quadrant_localization"] = maybe_build(
        "quadrant_localization", recs, dropped, "label",
        tier1_features(["top/left", "top/right", "bottom/left", "bottom/right"], {"target_fo_class": Value("string")}),
    )

    recs, dropped = build_closest_to_center(df, video_split)
    out["closest_to_center"] = maybe_build(
        "closest_to_center", recs, dropped, "label", tier1_features(FO_CLASSES),
    )

    recs, dropped = build_occlusion_state(df, video_split)
    occlusion_labels = ["fully_recognizable", "partially_obscured", "heavily_blood_covered", "mostly_out_of_sight", "heavily_blurred"]
    out["occlusion_state"] = maybe_build(
        "occlusion_state", recs, dropped, "label",
        tier1_features(occlusion_labels, {"target_fo_class": Value("string")}),
    )

    recs, dropped = build_sponge_saturation(df, video_split)
    out["sponge_saturation"] = maybe_build(
        "sponge_saturation", recs, dropped, "label",
        tier1_features(["clean", "partially_bloody", "fully_saturated"]),
    )

    recs, dropped = build_object_color(df, video_split)
    color_labels = sorted({r["label"] for r in recs}) or ["unknown"]
    out["object_color"] = maybe_build(
        "object_color", recs, dropped, "label",
        tier1_features(color_labels, {"target_fo_class": Value("string")}),
    )

    recs, dropped = build_grasped_by_instrument(df, video_split)
    out["grasped_by_instrument"] = maybe_build(
        "grasped_by_instrument", recs, dropped, "label",
        tier1_features(["no", "yes"], {"target_fo_class": Value("string")}),
    )

    recs, dropped = build_object_count(df, video_split)
    out["object_count"] = maybe_build(
        "object_count", recs, dropped, "label",
        tier1_features(["0", "1", "2", "3", "4+"], {"window_duration_s": Value("float32"), "count_type": Value("string")}),
    )

    # Tier 2 — image pair
    recs, dropped = build_instance_reidentification(df, video_split)
    out["instance_reidentification"] = maybe_build(
        "instance_reidentification", recs, dropped, "label", tier2_features(["no", "yes"]),
    )

    # Tier 3 — frame sequence
    recs, dropped = build_interaction_recognition(df, video_split)
    out["interaction_recognition"] = maybe_build(
        "interaction_recognition", recs, dropped, "label",
        tier3_features(["no", "yes"], {"event_time": Value("float32"), "action_type": Value("string")}),
    )

    recs, dropped = build_temporal_ordering_pairwise(df, video_split)
    out["temporal_ordering_pairwise"] = maybe_build(
        "temporal_ordering_pairwise", recs, dropped, "label", tier3_features(["no", "yes"]),
    )

    recs, dropped = build_fo_class_identification_window(df, video_split)
    out["fo_class_identification_window"] = maybe_build(
        "fo_class_identification_window", recs, dropped, "labels",
        tier3_multilabel_features(FO_CLASSES + ["none"]),
    )

    recs, dropped = build_co_occurrence_prediction(df, video_split)
    out["co_occurrence_prediction"] = maybe_build(
        "co_occurrence_prediction", recs, dropped, "labels",
        tier3_multilabel_features(FO_CLASSES + ["none"]),
    )

    recs, dropped = build_longest_duration_class(df, video_split)
    out["longest_duration_class"] = maybe_build(
        "longest_duration_class", recs, dropped, "label",
        tier3_features(FO_CLASSES + ["none"], with_target_fo_class=False),
    )

    recs, dropped = build_quadrant_localization_window(df, video_split)
    out["quadrant_localization_window"] = maybe_build(
        "quadrant_localization_window", recs, dropped, "label",
        tier3_features(QUADRANTS),  # with_target_fo_class=True by default, matches the record's target_fo_class key
    )

    recs, dropped = build_object_count_window(df, video_split)
    out["object_count_window"] = maybe_build(
        "object_count_window", recs, dropped, "label",
        tier3_features(["0", "1", "2", "3", "4+"], {"count_type": Value("string")}, with_target_fo_class=False),
    )

    return {k: v for k, v in out.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="path to qa_cache.parquet")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "probing_export", help="where to save_to_disk each config")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of videos held out as test, per procedure type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", action="store_true", help="also push every config to the Hub (requires --repo-id)")
    ap.add_argument("--repo-id", type=str, default=None, help="e.g. your-org/heico-vision-probing")
    ap.add_argument("--private", action="store_true", help="push as a private repo")
    args = ap.parse_args()

    if args.push and not args.repo_id:
        ap.error("--push requires --repo-id")

    df = pd.read_parquet(args.cache)
    video_split = make_video_split(df, test_frac=args.test_frac, seed=args.seed)

    n_train_videos = sum(1 for v in video_split.values() if v == "train")
    n_test_videos = sum(1 for v in video_split.values() if v == "test")
    print(f"Video-level split: {n_train_videos} train videos, {n_test_videos} test videos.")

    configs = build_all(df, video_split)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ds_dict in configs.items():
        path = args.out_dir / name
        ds_dict.save_to_disk(str(path))
        print(f"Saved {name} -> {path} (train={len(ds_dict['train'])}, test={len(ds_dict['test'])})")

    if args.push:
        for name, ds_dict in configs.items():
            print(f"Pushing {name} to {args.repo_id} ...")
            ds_dict.push_to_hub(args.repo_id, config_name=name, private=args.private)
        print("Push complete.")
    else:
        print("\n--push not set: nothing was uploaded to the Hub. "
              "Inspect ./probing_export/<task>/ locally, then re-run with "
              "--push --repo-id your-org/heico-vision-probing when ready.")


if __name__ == "__main__":
    main()
