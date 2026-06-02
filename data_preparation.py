# taken from https://github.com/IMSY-DKFZ/orena-focus/blob/main/examples/data_preparation.py
from focus import FocusConfig, download, set_config
from focus.preprocessing import FrameExtractorPreprocessor, VideoTimestampOverlayPreprocessor

DATASET = "heico"
DIR_DATA = "/projects/datasets_ML/orena/"
set_config(FocusConfig(root_dir=DIR_DATA))
# ── 2. Download ───────────────────────────────────────────────────────

# Downloads videos from HuggingFace Hub into <root>/heico/videos/.
# QA annotation parquet files are streamed on demand — no manual step needed.
# Safe to call repeatedly — skips the download if already complete.
download(DATASET)

# ── 3. Timestamp overlay (optional) ──────────────────────────────────

# Burns a visible hh:mm:ss counter into each video and writes the result to
# <root>/heico/overlayed/.  Skip this step if you do not need overlayed videos.
VideoTimestampOverlayPreprocessor().process(dataset=DATASET, max_workers=4)

# ── 4. Frame extraction ───────────────────────────────────────────────

# Extract every frame (stride=1) from the original videos into
# <root>/heico/frames/<video_stem>/frame{index:07d}.jpg.
FrameExtractorPreprocessor(stride=1).process(dataset=DATASET, max_workers=4)

# To extract from the overlayed videos instead, use a separate frames folder
# so both variants live side-by-side:
set_config(FocusConfig(root_dir=DIR_DATA, frames_folder="frames_overlay"))
FrameExtractorPreprocessor(stride=1, use_overlay=True).process(dataset=DATASET, max_workers=4)
set_config(FocusConfig(root_dir=DIR_DATA))  # restore default