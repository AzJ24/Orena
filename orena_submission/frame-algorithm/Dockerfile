FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS frame-algorithm-amd64

# torch 2.5.1 + torchvision are pre-installed in the base image (CUDA 12.4).
# requirements.txt only lists additional dependencies.

ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base image ships neither the shared libraries that
# opencv-python (a dependency of orena-focus) needs at import time, nor any
# compiler toolchain. Install system packages HERE, while the build still runs
# as root — everything below runs as the unprivileged 'user'. If one of your
# pip dependencies compiles from source (e.g. flash-attn), add build-essential.
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

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Fail the build if a dependency pulled in a torch built against a different CUDA
# major version than the platform's 12.4. Such a wheel does not error at runtime:
# torch.cuda.is_available() simply returns False and inference silently runs on
# CPU, hundreds of times slower. Catching it here needs no GPU.
RUN python -c "import torch, sys; \
    cu = torch.version.cuda; \
    print(f'torch {torch.__version__} | CUDA {cu}'); \
    sys.exit(0) if cu and cu.split('.')[0] == '12' else \
    sys.exit(f'FATAL: torch built for CUDA {cu}, platform provides 12.4')"

# ── Model + system prompt ──────────────────────────────────────────────────
# resources/frame_model/ holds the merged Qwen3.5-9B + LoRA-SFT checkpoint
# (combined-all-9b-8r-direct-ddp, checkpoint-938), a single ~18.8 GB
# model.safetensors — under the 50 GB single-layer limit, so no COPY splitting
# is needed. resources/prompts.py holds the exact 'direct' system prompt and
# answer parser the checkpoint was trained/evaluated with.
COPY --chown=user:user resources/ /opt/app/resources/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
