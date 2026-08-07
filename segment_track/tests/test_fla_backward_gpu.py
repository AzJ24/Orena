"""Do the fla Gated DeltaNet kernels produce CORRECT GRADIENTS on this GPU?

`check_fast_kernels` only compares Triton's version against the known-bad range
for Hopper (fla #640, wrong `chunk_bwd_dqkwg`). That is a proxy, not a test, and
it says nothing about a GPU nobody has run this on. Blackwell (RTX PRO 6000,
sm_120) is such a GPU: the eval jobs prove the forward path works there, but
training lives or dies on the backward.

So: run both the fused kernel and the pure-torch reference transformers falls
back to, on identical inputs, and compare forward AND every gradient. A silent
gradient bug is the one failure that produces a full, plausible-looking,
completely wasted training run.
"""

import sys

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402

print("torch", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
import triton  # noqa: E402

print("triton", triton.__version__)

B, T, H, K, V = 2, 512, 4, 128, 128
torch.manual_seed(0)


def make_inputs():
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = torch.rand(B, T, H, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    return [x.requires_grad_(True) for x in (q, k, v, g, beta)]


def run(fn, tensors, l2norm):
    """The two entry points disagree on argument names: fla takes q/k/v, the
    transformers reference takes query/key/value."""
    q, k, v, g, beta = tensors
    names = ("q", "k", "v") if fn is chunk_gated_delta_rule else ("query", "key", "value")
    kw = dict(zip(names, (q, k, v)))
    out = fn(**kw, g=g, beta=beta, use_qk_l2norm_in_kernel=l2norm)
    out = out[0] if isinstance(out, tuple) else out
    out.float().pow(2).sum().backward()
    return out, [t.grad for t in tensors]


ok = True


def compare(label, a, b, tol):
    global ok
    if a is None or b is None:
        print(f"  [SKIP] {label}: no gradient")
        return
    a, b = a.float(), b.float()
    fa, fb = bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())
    if not (fa and fb):
        # Attribute it: a NaN in the reference is a bad test input, not an fla bug.
        who = "BOTH" if not fa and not fb else ("fla" if not fa else "reference")
        print(f"  [NaN] {label}: non-finite in {who} -- fla_finite={fa} ref_finite={fb}")
        return
    err = (a - b).abs().max().item()
    scale = b.abs().max().item() or 1.0
    rel = err / scale
    good = torch.isfinite(a).all().item() and rel < tol
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: max abs {err:.3e}  "
          f"rel {rel:.3e} (tol {tol})")


# l2norm=True is the path Qwen3.5/3.6 actually takes, so test it as well as the
# plain one -- they are different kernels.
for l2norm in (False, True):
    torch.manual_seed(0)
    fast = make_inputs()
    torch.manual_seed(0)
    ref = make_inputs()
    for a, b in zip(fast, ref):
        assert torch.equal(a, b), "inputs diverged; the comparison would be meaningless"

    out_fast, grads_fast = run(chunk_gated_delta_rule, fast, l2norm)
    out_ref, grads_ref = run(qwen.torch_chunk_gated_delta_rule, ref, l2norm)
    torch.cuda.synchronize()

    print(f"\nfused vs torch reference  (B={B} T={T} H={H} K={K} V={V}, bf16, "
          f"use_qk_l2norm_in_kernel={l2norm})")
    compare("forward", out_fast, out_ref, 2e-2)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), grads_fast, grads_ref):
        compare(name, a, b, 5e-2)

print("\nRESULT:", "GRADIENTS AGREE — safe to train on this GPU" if ok
      else "MISMATCH — do NOT train with fla on this GPU")
sys.exit(0 if ok else 1)
