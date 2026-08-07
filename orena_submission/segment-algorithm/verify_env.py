"""Asserts the image's dependency stack is the one the checkpoint was validated on.

This exists because of a real failure on the FRAME track: torch was pinned in
requirements.txt against a base image built for a different CUDA, pip installed a
wheel for the wrong one, and nothing raised. `torch.cuda.is_available()` just
returned False, the model ran on CPU, and the loss was only visible after the image
had been built, uploaded and scored.

Nothing here needs a GPU, so `--build-time` runs inside `docker build` on the
GPU-less build node and fails the build instead of the submission. `--gpu` adds the
checks that only mean something on a real device and is run from the .sif test.

    python verify_env.py --build-time   # in the Dockerfile, no GPU
    python verify_env.py --gpu          # on gpu38, inside the built image
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys

# pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime. CUDA 12.8 is REQUIRED, not
# preferred: the challenge's GPU for models this size is an RTX PRO 6000 Blackwell
# (sm_120), and CUDA 12.4 -- the template's base, and what the first submission
# shipped -- has no kernels past sm_90. It fails inside from_pretrained with
# "no kernel image is available for execution on the device". 12.8 also covers
# sm_90, so this image is correct on H100/H200 too.
EXPECTED_TORCH = "2.11.0"
EXPECTED_CUDA_MAJOR = "12"
EXPECTED_CUDA = "12.8"  # the base image tag's CUDA; warn (not fail) on a 12.x drift
EXPECTED_PACKAGES = {
    "transformers": "5.9.0",
    "accelerate": "1.14.0",
    "orena-focus": "0.3.5",
    "decord": "0.6.0",
}
# Deliberately absent. On torch 2.11 fla would now WORK (triton >= 3.3) and would cut
# ~6.1 s/question to ~2.9 s -- but the budget is 15 s and the margin is already wide,
# so the dependency is not worth reintroducing. Asserting its absence keeps the
# shipped environment identical to the one measured on Blackwell.
FORBIDDEN_PACKAGES = ("flash-linear-attention", "fla-core")
# focus.foreign_objects.FOType.names() as baked into the trained system prompt.
EXPECTED_FO_CLASSES = [
    "Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain",
    "Needle", "Gallstone", "Specimen", "Mesh", "Absorbable Hemostatic Agent",
]

failures: list[str] = []
warnings: list[str] = []


def check(ok: bool, label: str, detail: str = "", fail_detail: str = "") -> None:
    """`detail` is shown either way; `fail_detail` only when the check fails."""
    note = detail if ok else (fail_detail or detail)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' — ' + note) if note else ''}")
    if not ok:
        failures.append(label)


def warn(ok: bool, label: str, detail: str = "", fail_detail: str = "") -> None:
    note = detail if ok else (fail_detail or detail)
    print(f"  [{'OK ' if ok else 'WARN'}] {label}{(' — ' + note) if note else ''}")
    if not ok:
        warnings.append(label)


def check_torch() -> None:
    import torch

    print("\n== torch / CUDA ==")
    print(f"  torch {torch.__version__} | CUDA build {torch.version.cuda}")
    print(f"  loaded from {torch.__file__}")

    # THE check. A torch built for the wrong CUDA major does not raise at import
    # or at load; it silently reports no GPU.
    major = (torch.version.cuda or "").split(".")[0]
    check(major == EXPECTED_CUDA_MAJOR, "torch CUDA major is 12",
          fail_detail=f"built for CUDA {torch.version.cuda}, base image provides "
                      f"{EXPECTED_CUDA} — this is the silent-CPU-fallback bug")

    version = torch.__version__.split("+")[0]
    check(version == EXPECTED_TORCH, f"torch version is {EXPECTED_TORCH}", f"got {version}")
    warn(torch.version.cuda == EXPECTED_CUDA, f"torch CUDA is exactly {EXPECTED_CUDA}",
         f"got {torch.version.cuda}")

    # If pip replaced the base image's torch, the loaded copy sits in the --user
    # site-packages instead of the environment's. That is precisely how a pinned
    # torch overrides the base image, so name it rather than infer it later.
    in_user_site = "/.local/" in torch.__file__
    check(not in_user_site, "torch comes from the base image, not pip --user",
          fail_detail="a pip-installed torch is shadowing the base image's")

    import torchvision
    # torchvision reports CUDA as "12060"-style, torch as "12.4"; compare majors.
    tv_cuda = getattr(torchvision.version, "cuda", None)
    tv_major = str(tv_cuda).replace(".", "")[:2] if tv_cuda else None
    warn(tv_major in (None, str(major)), "torchvision CUDA major matches torch",
         f"torchvision {torchvision.__version__} (CUDA {tv_cuda})")


def check_packages() -> None:
    print("\n== pinned packages ==")
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            check(False, f"{name} installed", "NOT INSTALLED")
            continue
        check(got == expected, f"{name} == {expected}", f"got {got}")


def check_fla() -> None:
    print("\n== linear-attention kernels (48 of the model's layers) ==")
    for name in FORBIDDEN_PACKAGES:
        try:
            got = md.version(name)
            installed = True
        except md.PackageNotFoundError:
            got, installed = "", False
        # Not fatal on this base image (torch 2.11 satisfies fla's triton >= 3.3), but
        # its presence would mean the shipped environment differs from the one
        # measured on Blackwell. Kept absent on purpose; see requirements.txt.
        check(not installed, f"{name} is absent (kept out deliberately; see requirements.txt)",
              fail_detail=f"{name}=={got} installed; shipped env no longer matches what was measured")

    # With fla gone, transformers substitutes its own torch implementation. That is
    # the intended path here, so confirm it is wired up rather than treating the
    # "fast path is not available" warning as a problem.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    check(m.torch_chunk_gated_delta_rule is not None,
          "transformers' torch Gated DeltaNet fallback is available")
    print("  [note] 'fast path is not available' at load is EXPECTED — 6.1 s/question "
          "measured on Blackwell vs ~2.9 s with fused kernels, against a 15 s budget")


def check_model_stack() -> None:
    print("\n== model + prompt stack ==")
    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: F401
        check(True, "Qwen3_5ForConditionalGeneration importable")
    except Exception as exc:
        check(False, "Qwen3_5ForConditionalGeneration importable", repr(exc))

    try:
        import decord  # noqa: F401
        check(True, "decord importable")
    except Exception as exc:
        check(False, "decord importable", repr(exc))

    try:
        from focus.foreign_objects import FOType
        names = list(FOType.names())
        check(names == EXPECTED_FO_CLASSES, "FO class registry matches training",
              f"got {names}" if names != EXPECTED_FO_CLASSES else f"{len(names)} classes")
    except Exception as exc:
        check(False, "FO class registry matches training", repr(exc))


def check_gpu() -> None:
    import torch

    print("\n== GPU (real device) ==")
    check(torch.cuda.is_available(), "torch.cuda.is_available()",
          "THE silent-failure symptom: wrong CUDA build, or no --nv / --gres")
    if not torch.cuda.is_available():
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    arches = torch.cuda.get_arch_list()
    print(f"  {name} | sm_{cap[0]}{cap[1]} | {total:.1f} GiB")
    print(f"  torch was compiled for: {' '.join(arches)}")

    # THE check this file previously missed. `cuda.is_available()` returns True on a
    # GPU this torch has no kernels for -- an sm_120 Blackwell reports available,
    # then dies at the first launch with "no kernel image is available for execution
    # on the device", partway through loading the model. Asserting the device's
    # capability is actually in the compiled arch list catches it here instead.
    check(f"sm_{cap[0]}{cap[1]}" in arches,
          f"torch has kernels for this device (sm_{cap[0]}{cap[1]})",
          fail_detail=f"compiled for {arches} — every kernel launch will fail; "
                      f"sm_120 (Blackwell) needs CUDA >= 12.8")

    # And prove it, rather than trusting the arch list: a real launch on the device.
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        ok = bool(torch.isfinite((x @ x).float().sum()).item())
    except RuntimeError as exc:
        ok = False
        print(f"       launch raised: {exc}".split("\n")[0])
    check(ok, "bf16 matmul actually launches on this device")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-time", action="store_true",
                    help="checks that need no GPU (run inside docker build)")
    ap.add_argument("--gpu", action="store_true", help="add real-device checks")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}")
    check_torch()
    check_packages()
    check_fla()
    check_model_stack()
    if args.gpu:
        check_gpu()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    if warnings:
        print(f"PASSED with {len(warnings)} warning(s): " + "; ".join(warnings))
    else:
        print("PASSED — dependency stack matches the validated environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
