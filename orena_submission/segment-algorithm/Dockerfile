# ── Stage 1: dependencies ─────────────────────────────────────────────────────
# Split from the weights on purpose. `preflight_deps.sh` builds ONLY this stage,
# from a context holding two small files, and proves the torch/CUDA/fla stack in
# minutes — before an hour goes into copying 55 GB of weights that a mis-resolved
# dependency would have wasted.
FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS deps

# torch 2.11.0 + torchvision are pre-installed in the base image (CUDA 12.8).
# requirements.txt only lists additional dependencies.
#
# CUDA 12.8 is REQUIRED, not preferred. The challenge lets you pick the evaluation
# GPU, and a 27B model at 55 GiB does not fit the L40S (48 GB) -- which leaves the
# RTX PRO 6000 Blackwell (96 GB), and Blackwell is sm_120. The template's
# 2.5.1-cuda12.4 base compiles kernels only up to sm_90, so it loads the weights and
# then dies inside from_pretrained with "no kernel image is available for execution
# on the device". That is what failed the first submission; reproduced on gpu38's
# Blackwell card, and fixed here.
#
# torch 2.11.0+cu128 compiles for sm_75/80/86/90/100/120, so this image is also
# correct on H100/H200 if the platform's hardware changes again.
#
# Verified on an RTX PRO 6000 Blackwell with the merged 27B checkpoint over 20 real
# TEST questions: setup 14.6 s, 20/20 answered, mean 6.1 s/question (max 10.1 s
# against a 15 s budget), peak VRAM 55.1 GiB of 96 GB.

ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base image ships neither the shared libraries that
# opencv-python (a dependency of orena-focus) needs at import time, nor any
# compiler toolchain. Install system packages HERE, while the build still runs
# as root — everything below runs as the unprivileged 'user'. No compiler is
# needed: flash-linear-attention is pure Python over Triton, and decord ships a
# manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# --break-system-packages is required because this base image ships Debian's system
# Python 3.12 (the older conda-based pytorch images did not) and marks it PEP 668
# externally managed. It does not weaken the isolation --user gives: packages still
# land in /home/user/.local/lib/python3.12/site-packages, which precedes the base
# image's /usr/local/lib/python3.12/dist-packages on sys.path, so torch is never
# touched. verify_env.py asserts exactly that.
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --break-system-packages \
    --requirement /opt/app/requirements.txt

# Fail the BUILD rather than the submission. verify_env.py asserts that the torch
# the image ends up with is still the base image's CUDA 12.8 build — the exact
# failure that is invisible at runtime, because a torch built for the wrong CUDA
# does not raise: torch.cuda.is_available() simply returns False and a 27B model
# runs on CPU. It also pins the dependency versions and checks that the FO class
# list the system prompt is built from still matches training. Needs no GPU.
COPY --chown=user:user verify_env.py /opt/app/
RUN python /opt/app/verify_env.py --build-time

# ── Stage 2: the submission image ─────────────────────────────────────────────
FROM deps AS segment-algorithm-amd64

# resources/segment_model/ holds the merged Qwen3.6-27B + LoRA-r8 SFT checkpoint
# (segment-27b-alldata-n80-650-20260730), ~54.7 GB of bf16 safetensors.
#
# ECR layer-size limit: a single image layer must not exceed 50 GB, and each COPY
# instruction produces exactly one layer. The checkpoint as merged had a 49.83 GB
# shard — inside the limit by 0.35%, which is not a margin worth a 55 GB upload.
# reshard_weights.py rewrites it into ~8 GB shards; the two globs below then split
# them across two layers of roughly 32 GB and 23 GB. The globs cover up to nine
# shards, which 8 GB shards of a 27B bf16 model cannot exceed.
COPY --chown=user:user resources/segment_model/model-0000[1-4]-of-*.safetensors /opt/app/resources/segment_model/
COPY --chown=user:user resources/segment_model/model-0000[5-9]-of-*.safetensors /opt/app/resources/segment_model/

# Config, tokenizer, chat template and the weight index — a few MB in their own layer.
COPY --chown=user:user resources/segment_model/*.json /opt/app/resources/segment_model/
COPY --chown=user:user resources/segment_model/*.jinja /opt/app/resources/segment_model/

# The system prompt + answer parser the checkpoint was trained with, and the clip
# reconstruction that rebuilds training's frame sampling from a platform clip.
COPY --chown=user:user resources/prompts.py resources/clip_frames.py /opt/app/resources/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
