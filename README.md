# Orena — FOCUS Challenge (Frame & Segment Track)

Dataset: [HEICO](https://github.com/IMSY-DKFZ/orena-focus) surgical video QA.

## Data location

```
/projects/datasets_ML/orena/
└── heico/
    ├── videos/          # source .avi files (downloaded)
    ├── frames/          # extracted JPEGs — frame{index:07d}.jpg per video
    ├── frames_overlay/  # same with burned-in timestamp overlay
    └── overlayed/       # overlayed .avi files
```

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Scripts

| File | Purpose |
|---|---|
| `data_preparation.py` | Download videos + extract frames (run once) |
| `data_exploration.ipynb` | Interactive notebook for data exploration |
