"""Does the fla Triton kernel actually compile and run on this GPU?"""
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("imported chunk_gated_delta_rule OK")

B, T, H, K, V = 1, 256, 4, 128, 128
dt = torch.bfloat16
q = torch.randn(B, T, H, K, device="cuda", dtype=dt)
k = torch.randn(B, T, H, K, device="cuda", dtype=dt)
v = torch.randn(B, T, H, V, device="cuda", dtype=dt)
g = torch.rand(B, T, H, device="cuda", dtype=torch.float32).log()
beta = torch.rand(B, T, H, device="cuda", dtype=dt)
out, state = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta)
torch.cuda.synchronize()
print("KERNEL RAN. out", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
