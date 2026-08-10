"""Checks the procedure sampler against the real data distribution. CPU only, no disk.

This is the module the whole track hangs on: it runs once at export time and again
inside the submission container, and any disagreement between the two is a silent
train/inference mismatch. Everything here is asserted against an independently computed
expectation, never against the sampler's own output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling import (  # noqa: E402
    DEFAULT_N_MAX, FPS, TARGET_PERIOD_S, describe, parse_question_timestamps,
    sample_indices_5fps, to_local_index, to_native_index,
)

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


print("1. index-space conversions")
# heico 25 fps and lapchole 30 fps must land on the SAME wall-clock time.
check("heico native index", to_native_index(2775, 25.0) == 13875)
check("lapchole native index", to_native_index(2775, 30.0) == 16650)
check("both are 555.0 s", 13875 / 25.0 == 16650 / 30.0 == 2775 / FPS == 555.0)
check("local index for a procedure clip (start=0) is the identity",
      to_local_index(2775, 0.0) == 2775)
check("local index shifts by the clip start", to_local_index(2775, 555.0) == 0)

print("\n2. timestamp parsing")
q = "There is one Sponge in the frame at 00:09:19. When is it retrieved?"
check("single anchor", parse_question_timestamps(q) == [559.0], str(parse_question_timestamps(q)))
q2 = "What types of foreign objects are seen between 01:49:00 and 02:23:01?"
check("two anchors", parse_question_timestamps(q2) == [6540.0, 8581.0])
check("no anchor", parse_question_timestamps("How many Clips are inserted?") == [])
check("the window line would be parsed if we let it",
      len(parse_question_timestamps("Procedure observed from 00:00:00 to 02:47:31.")) == 2,
      "so the builder MUST parse the raw question first")

print("\n3. short window -> TRUE 5 s density, far below the cap")
idx = sample_indices_5fps(0.0, 1579.0, "How many Clips?", n_max=DEFAULT_N_MAX)
d = describe(idx)
check("well under the cap", len(idx) < DEFAULT_N_MAX, f"{len(idx)} of {DEFAULT_N_MAX}")
check("every gap at the 5 s target", d["at_target"] == 1.0,
      f"max gap {d['max_gap_s']:.1f}s")
check("count is even", len(idx) % 2 == 0)
check("spans the window", idx[0] == 0 and abs(idx[-1] / FPS - 1579.0) < TARGET_PERIOD_S,
      f"{idx[0]/FPS:.0f}s..{idx[-1]/FPS:.0f}s")

print("\n4. long window -> capped, stretched uniformly")
long_end = 17780.0                      # the longest procedure window, 4 h 56 m
idx = sample_indices_5fps(0.0, long_end, "How many Clips?", n_max=DEFAULT_N_MAX)
d = describe(idx)
check("hits the cap", len(idx) == DEFAULT_N_MAX, str(len(idx)))
check("count is even", len(idx) % 2 == 0)
check("uniform, no 5 s density", d["at_target"] == 0.0, f"gaps ~{d['max_gap_s']:.0f}s")
check("spacing matches duration/n_max", abs(d["max_gap_s"] - long_end / DEFAULT_N_MAX) < 1.0,
      f"{d['max_gap_s']:.1f}s vs {long_end/DEFAULT_N_MAX:.1f}s")

print("\n5. long window + anchored question -> dense where the question points")
anchor_s = 6540.0                       # 01:49:00
q_anchored = f"There is one Sponge in the frame at 01:49:00. When is it retrieved?"
idx_a = sample_indices_5fps(0.0, long_end, q_anchored, n_max=DEFAULT_N_MAX)
check("still capped and even", len(idx_a) <= DEFAULT_N_MAX and len(idx_a) % 2 == 0,
      str(len(idx_a)))
check("monotone", all(b > a for a, b in zip(idx_a, idx_a[1:])))

near = [i for i in idx_a if abs(i / FPS - anchor_s) < 300]
near_plain = [i for i in idx if abs(i / FPS - anchor_s) < 300]
check("far more frames near the anchor than uniform gives",
      len(near) > 3 * len(near_plain), f"{len(near)} vs {len(near_plain)} in +-300 s")
gaps_near = [(b - a) / FPS for a, b in zip(near, near[1:])]
check("local spacing reaches the 5 s target",
      gaps_near and min(gaps_near) <= TARGET_PERIOD_S + 1e-6,
      f"min local gap {min(gaps_near):.1f}s" if gaps_near else "no local frames")
check("the rest of the procedure is still covered",
      idx_a[0] < 100 and idx_a[-1] > 0.9 * long_end * FPS,
      f"{idx_a[0]/FPS:.0f}s..{idx_a[-1]/FPS:.0f}s")

print("\n6. two-anchor question spans the sub-window")
idx_b = sample_indices_5fps(0.0, long_end, q2, n_max=DEFAULT_N_MAX)
inside = [i for i in idx_b if 6540.0 <= i / FPS <= 8581.0]
inside_plain = [i for i in idx if 6540.0 <= i / FPS <= 8581.0]
check("dense across [T1, T2]", len(inside) > 2 * len(inside_plain),
      f"{len(inside)} vs {len(inside_plain)}")

print("\n7. anchoring is a no-op when the window already fits at 5 s")
short_q = sample_indices_5fps(0.0, 1579.0, q, n_max=DEFAULT_N_MAX)
short_p = sample_indices_5fps(0.0, 1579.0, "How many Clips?", n_max=DEFAULT_N_MAX)
check("identical to the unanchored grid", short_q == short_p,
      "already at target density, so there is nothing to buy")

print("\n8. determinism and edge cases")
check("same inputs -> same indices",
      sample_indices_5fps(0.0, long_end, q_anchored) == idx_a)
check("an out-of-window anchor is ignored",
      sample_indices_5fps(0.0, 600.0, "at 09:99:99 what happens?", n_max=64)
      == sample_indices_5fps(0.0, 600.0, None, n_max=64))
check("degenerate window survives", len(sample_indices_5fps(0.0, 0.0)) == 2)
for bad, why in [((0.0, -1.0, None, 64), "end before start"),
                 ((0.0, 100.0, None, 63), "odd n_max")]:
    try:
        sample_indices_5fps(*bad)
        check(f"rejects {why}", False, "no error raised")
    except ValueError:
        check(f"rejects {why}", True)

print("\n9. the cap holds across the real duration distribution")
for minutes in (5, 21, 26, 43, 64, 100, 190, 296):
    got = sample_indices_5fps(0.0, minutes * 60.0, "How many Clips?", n_max=DEFAULT_N_MAX)
    dd = describe(got)
    at5 = dd["at_target"] == 1.0
    check(f"{minutes:3d} min -> {len(got):3d} frames, gaps <= {dd['max_gap_s']:5.1f}s",
          len(got) <= DEFAULT_N_MAX and len(got) % 2 == 0,
          "all at 5 s" if at5 else "stretched")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
