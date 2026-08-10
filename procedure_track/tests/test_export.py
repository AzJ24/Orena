"""Validates the exported procedure JSONL against the source parquet and real disk."""

import collections
import glob
import json
import statistics
import sys
from pathlib import Path

PROC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROC))
sys.path.insert(0, str(PROC.parent / "segment_track"))

import pandas as pd  # noqa: E402

from clip_sampling import frame_file  # noqa: E402

from sampling import DEFAULT_N_MAX, FPS, TARGET_PERIOD_S, to_native_index  # noqa: E402

EXPORT = PROC / "sft_export"
HUB = Path.home() / ".cache/huggingface/hub"

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def load(name):
    return [json.loads(line) for line in (EXPORT / f"{name}.jsonl").open()]


def parquet(ds, split):
    hits = sorted(glob.glob(str(
        HUB / f"datasets--orena-dkfz--{ds}-focus-vqa/snapshots/*/data/procedure/{split}.parquet")))
    return pd.read_parquet(hits[-1])


for name in ("train", "eval", "test"):
    if not (EXPORT / f"{name}.jsonl").exists():
        print(f"{name}.jsonl missing — run build_procedure_sft_dataset.py first")
        sys.exit(1)

train, ev, test = load("train"), load("eval"), load("test")

print("1. row counts reconcile with the source parquet")
src_train = sum(len(parquet(d, "train")) for d in ("heico", "lapchole"))
src_test = sum(len(parquet(d, "test")) for d in ("heico", "lapchole"))
check("train + eval == parquet train", len(train) + len(ev) == src_train,
      f"{len(train)}+{len(ev)}={len(train)+len(ev)} vs {src_train}")
check("test == parquet test", len(test) == src_test, f"{len(test)} vs {src_test}")
check("eval is ~10% of train rows", 0.05 < len(ev) / (len(train) + len(ev)) < 0.20,
      f"{100*len(ev)/(len(train)+len(ev)):.1f}%")

print("\n2. splits are video-disjoint (prefixes of one video overlap totally)")
vtr, vev, vte = ({r["videoID"] for r in s} for s in (train, ev, test))
check("train n eval empty", not (vtr & vev), f"{len(vtr)} train / {len(vev)} eval videos")
check("train n test empty", not (vtr & vte))
check("eval n test empty", not (vev & vte))
rows = train + ev + test
check("uid is globally unique", len({r["uid"] for r in rows}) == len(rows))
for ds in ("heico", "lapchole"):
    sub = [r["qID"] for r in rows if r["source_dataset"] == ds]
    check(f"qID unique within {ds}", len(set(sub)) == len(sub), f"{len(sub)} rows")

print("\n3. the window really is a prefix")
check("every window starts at 0", all(r["start_time"] == 0 for r in rows))
ends = collections.Counter()
for r in rows:
    ends[r["videoID"]] += 1
check("each video appears under many windows",
      statistics.median(ends.values()) > 10, f"median {statistics.median(ends.values()):.0f}")

print("\n4. sampling policy")
n = [r["n_frames"] for r in rows]
check("all even (temporal_patch_size=2)", all(x % 2 == 0 for x in n))
check("all within the cap", max(n) <= DEFAULT_N_MAX, f"max {max(n)}")
check("counts VARY — the capped-5s policy is active", len(set(n)) > 10,
      f"{len(set(n))} distinct counts, median {statistics.median(n):.0f}")
at5 = [r["gap_at_target"] for r in rows]
frac = sum(x >= 0.999 for x in at5) / len(at5)
check("a large minority reach true 5 s density", 0.3 < frac < 0.9,
      f"{100*frac:.1f}% of rows (plan.md §2.2 predicts 56.6% at n_max=768)")
check("indices monotone and inside the window", all(
    all(b > a for a, b in zip(r["frames_indices"], r["frames_indices"][1:]))
    and r["frames_indices"][0] >= 0
    and r["frames_indices"][-1] <= round(r["end_time"] * FPS) + 1
    for r in rows[:500]))
check("max gap never exceeds the uniform fallback", all(
    r["gap_max_gap_s"] <= max(TARGET_PERIOD_S, r["duration"] / DEFAULT_N_MAX) * 2.5
    for r in rows), f"worst {max(r['gap_max_gap_s'] for r in rows):.0f}s")

print("\n5. anchored rows really are denser where the question points")
anch = [r for r in rows if r["n_quoted_timestamps"] > 0 and r["gap_at_target"] < 0.999]
plain = [r for r in rows if r["n_quoted_timestamps"] == 0 and r["gap_at_target"] < 0.999]
check("both populations exist", bool(anch) and bool(plain),
      f"{len(anch)} anchored / {len(plain)} plain (stretched rows only)")
if anch and plain:
    check("anchored rows have a much finer minimum gap",
          statistics.median(r["gap_min_gap_s"] for r in anch)
          < statistics.median(r["gap_min_gap_s"] for r in plain),
          f"{statistics.median(r['gap_min_gap_s'] for r in anch):.1f}s vs "
          f"{statistics.median(r['gap_min_gap_s'] for r in plain):.1f}s")

print("\n6. prompt assembly")
s = train[0]
q = s["messages"][0]["content"][1]["text"]
check("window line prepended", q.startswith("Procedure observed from 00:00:00 to "), q[:60])
check("window line matches end_time",
      f"{int(s['end_time'])//3600:02d}:{(int(s['end_time'])%3600)//60:02d}" in q)
check("video placeholder present", s["messages"][0]["content"][0] == {"type": "video"})
check("answer is a bare string", isinstance(s["messages"][1]["content"][0]["text"], str))

print("\n7. frames resolve on disk")
for r in rows[:20]:
    p = frame_file(r["frame_dir"], to_native_index(r["frames_indices"][-1], r["base_fps"]))
    if not p.exists():
        check("last frame of every sampled row exists", False, str(p))
        break
else:
    check("last frame of every sampled row exists", True, "checked 20 rows")
check("overlay frames are used", all("frames_overlay" in r["frame_dir"] for r in rows[:100]),
      rows[0]["frame_dir"])

print("\n8. distributions match the source data")
fmt = collections.Counter(r["format"] for r in train + ev)
share = {k: v / sum(fmt.values()) for k, v in fmt.items()}
check("time is ~36% of train", 0.32 < share.get("time", 0) < 0.40, f"{100*share.get('time',0):.1f}%")
check("number is ~21% of train", 0.17 < share.get("number", 0) < 0.26,
      f"{100*share.get('number',0):.1f}%")
check("heico test is the OOD procedure",
      {r["procedure_type"] for r in test if r["source_dataset"] == "heico"}
      == {"Sigmoid Resection"})

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
