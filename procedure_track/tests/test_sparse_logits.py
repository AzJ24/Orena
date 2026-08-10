"""Proves the sparse-logits loss equals the full-logits loss. CPU only, no model.

Uses a toy lm_head and random hidden states, so the arithmetic is checked against an
independent reference rather than against the thing it replaces. If this drifts, the
training loss is silently wrong -- which is the one failure that produces a model that
trains, converges, and is useless.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sparse_logits_loss import sparse_causal_loss, supervised_positions  # noqa: E402

torch.manual_seed(0)
ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


B, L, H, V = 2, 64, 32, 500
lm_head = torch.nn.Linear(H, V, bias=False)
hidden = torch.randn(B, L, H)


class ToyModel(torch.nn.Module):
    """Stands in for Qwen3_5ForConditionalGeneration's `logits_to_keep` contract:
    an int slices the tail, a tensor selects positions, both before lm_head."""

    def forward(self, hidden_states=None, logits_to_keep=0, **kw):
        idx = (slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)
               else logits_to_keep)
        out = type("O", (), {})()
        out.logits = lm_head(hidden_states[:, idx, :])
        return out


model = ToyModel()

# The real shape: a long masked prompt, a handful of supervised answer tokens at the end.
labels = torch.full((B, L), -100)
labels[0, 55:62] = torch.randint(0, V, (7,))
labels[1, 58:62] = torch.randint(0, V, (4,))

print("1. position selection")
keep = supervised_positions(labels)
check("only supervised positions kept", keep.numel() == 7, f"{keep.numel()} of {L-1}")
check("union across the batch", keep.min().item() == 54 and keep.max().item() == 60,
      f"{keep.min().item()}..{keep.max().item()}")
check("a label at t reads the hidden state at t-1",
      all(labels[:, k + 1].ne(-100).any() for k in keep.tolist()))

print("\n2. loss equals the full-logits reference")
full_logits = lm_head(hidden).float()
ref_mean = F.cross_entropy(full_logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                           ignore_index=-100)
got_mean, _ = sparse_causal_loss(model, {"hidden_states": hidden}, labels)
check("mean reduction matches", torch.allclose(ref_mean, got_mean, atol=1e-6),
      f"{ref_mean.item():.8f} vs {got_mean.item():.8f}")

n_items = int((labels[:, 1:] != -100).sum())
ref_sum = F.cross_entropy(full_logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                          ignore_index=-100, reduction="sum") / n_items
got_sum, _ = sparse_causal_loss(model, {"hidden_states": hidden}, labels,
                                num_items_in_batch=n_items)
check("sum/num_items reduction matches", torch.allclose(ref_sum, got_sum, atol=1e-6),
      f"{ref_sum.item():.8f} vs {got_sum.item():.8f}")

print("\n3. gradients match too")
hid_a = hidden.clone().requires_grad_(True)
F.cross_entropy(lm_head(hid_a)[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1),
                ignore_index=-100).backward()
hid_b = hidden.clone().requires_grad_(True)
sparse_causal_loss(model, {"hidden_states": hid_b}, labels)[0].backward()
check("hidden-state grads identical", torch.allclose(hid_a.grad, hid_b.grad, atol=1e-6),
      f"max |delta| = {(hid_a.grad - hid_b.grad).abs().max().item():.2e}")
check("grad is zero on unsupervised positions", hid_a.grad[:, :53].abs().max() == 0,
      "so the discarded logits really did contribute nothing")

print("\n4. memory: what this actually saves")
saved = (L - 1 - keep.numel()) / (L - 1)
check("logit rows avoided", saved > 0.85, f"{100*saved:.1f}% at L={L}")
real_L, real_V = 66_300, 248_320
full_gb = real_L * real_V * 2 / 1e9
print(f"      at the real shape ({real_L:,} tokens x {real_V:,} vocab): "
      f"{full_gb:.1f} GB bf16 + {2*full_gb:.1f} GB fp32 -> "
      f"{10 * real_V * 6 / 1e6:.0f} MB")

print("\n5. edge cases")
empty = torch.full((B, L), -100)
loss, out = sparse_causal_loss(model, {"hidden_states": hidden}, empty)
check("all-masked batch returns 0, not NaN", loss.item() == 0.0 and out is None)
one = torch.full((1, L), -100)
one[0, L - 1] = 3
loss1, _ = sparse_causal_loss(model, {"hidden_states": hidden[:1]}, one)
check("single supervised token works", torch.isfinite(loss1), f"{loss1.item():.4f}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
