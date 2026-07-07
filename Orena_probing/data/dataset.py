from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset, load_from_disk

SINGLE_FRAME_TASKS = [
    "fo_class_identification",
    "fo_class_identification_multilabel",
    "quadrant_localization",
    "closest_to_center",
    "grasped_by_instrument",
    "object_count",
]

# On hold — not wired up to extract_features.py / train_probe.py yet.
PAIR_TASKS = ["instance_reidentification"]
WINDOW_TASKS = [
    "fo_class_identification_window",
    "quadrant_localization_window",
    "object_count_window",
    "interaction_recognition",
    "temporal_ordering_pairwise",
    "co_occurrence_prediction",
    "longest_duration_class",
]


def task_input_type(task: str) -> str:
    if task in SINGLE_FRAME_TASKS:
        return "single_frame"
    if task in PAIR_TASKS:
        return "pair"
    if task in WINDOW_TASKS:
        return "window"
    raise ValueError(f"Unknown task '{task}'")


@dataclass
class LabelInfo:
    label_key: str          # "label" or "labels"
    multi_label: bool
    num_classes: int
    class_names: list[str]


def get_label_info(ds: Dataset) -> LabelInfo:
    if "label" in ds.features:
        label_key = "label"
        feature = ds.features["label"]
        return LabelInfo(label_key, multi_label=False, num_classes=len(feature.names), class_names=feature.names)
    if "labels" in ds.features:
        label_key = "labels"
        feature = ds.features["labels"].feature  # List(ClassLabel) -> ClassLabel
        return LabelInfo(label_key, multi_label=True, num_classes=len(feature.names), class_names=feature.names)
    raise ValueError("Dataset has neither a 'label' nor a 'labels' column")


def load_task(
    task: str,
    split: str,
    source: str = "local",
    local_dir: str = "../probing_export",
    hub_repo: str = "Machine-Learning-Oncology/orena_probing",
) -> Dataset:
    """Load one probing task/split, from the local probing_export/ export or the HF Hub."""
    if source == "local":
        return load_from_disk(f"{local_dir}/{task}")[split]
    if source == "hub":
        return load_dataset(hub_repo, task, split=split)
    raise ValueError(f"Unknown source '{source}', expected 'local' or 'hub'")
