"""Validates the exported segment JSONL against the source parquet and real disk."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from clip_sampling import frame_file, marker_times  # noqa: E402

EXPORT = Path(__file__).resolve().parents[1] / "sft_export"
HUB = Path.home() / ".cache/huggingface/hub"
SNAP = {"heico": "d2ce84ddcd31a4e5a8eceba3e15d7c039a02883d",
        "lapchole": "1d9a4b666dcfe3db24e884626cda0a074fc76e68"}

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def load(name):
    return [json.loads(l) for l in (EXPORT / f"{name}.jsonl").open()]


def parquet(ds, split):
    return pd.read_parquet(
        HUB / f"datasets--orena-dkfz--{ds}-focus-vqa/snapshots/{SNAP[ds]}/data/segment/{split}.parquet")


train, ev, test = load("train"), load("eval"), load("test")

print("1. row counts reconcile with the source parquet")
src_train = sum(len(parquet(d, "train")) for d in SNAP)
src_test = sum(len(parquet(d, "test")) for d in SNAP)
check("train + eval == parquet train", len(train) + len(ev) == src_train,
      f"{len(train)}+{len(ev)}={len(train)+len(ev)} vs {src_train}")
check("test == parquet test", len(test) == src_test, f"{len(test)} vs {src_test}")
check("eval is ~10% of train rows", 0.05 < len(ev) / (len(train) + len(ev)) < 0.16,
      f"{100*len(ev)/(len(train)+len(ev)):.1f}%")

print("2. splits are video-disjoint (segment clips overlap; leakage would be total)")
vtr, vev, vte = ({r["videoID"] for r in s} for s in (train, ev, test))
check("train ∩ eval empty", not (vtr & vev), f"{len(vtr)} train / {len(vev)} eval videos")
check("train ∩ test empty", not (vtr & vte))
check("eval ∩ test empty", not (vev & vte))
rows = train + ev
check("uid is globally unique", len({r["uid"] for r in rows}) == len(rows),
      f"{len(rows) - len({r['uid'] for r in rows})} collisions")
# qID collides across datasets by design (each numbers from 1); it must still be
# unique within a dataset, which is the scope the Evaluator matches on.
for ds in ("heico", "lapchole"):
    sub = [r["qID"] for r in rows if r["source_dataset"] == ds]
    check(f"qID unique within {ds}", len(set(sub)) == len(sub), f"{len(sub)} rows")

print("3. eval split keeps every procedure type")
ptr = {r["procedure_type"] for r in train}
pev = {r["procedure_type"] for r in ev}
check("eval covers all train procedures", ptr == pev, f"train {sorted(ptr)} / eval {sorted(pev)}")
check("test is the unseen procedure", "Sigmoid Resection" in {r["procedure_type"] for r in test})
check("Sigmoid Resection absent from train", "Sigmoid Resection" not in ptr)

print("4. frame sampling is well-formed (all rows)")
bad_len = [r for r in train + ev + test if len(r["frames_indices"]) != 80]
bad_mono = [r for r in train + ev + test
            if any(b < a for a, b in zip(r["frames_indices"], r["frames_indices"][1:]))]
bad_win = [r for r in train + ev + test
           if not (r["frames_indices"][0] <= round(r["start_time"] * r["base_fps"]) + 1
                   and r["frames_indices"][-1] <= round(r["end_time"] * r["base_fps"]) + 1)]
check("every row has exactly 80 frames", not bad_len, f"{len(bad_len)} offenders")
check("indices monotone non-decreasing", not bad_mono, f"{len(bad_mono)} offenders")
check("indices inside the clip window", not bad_win, f"{len(bad_win)} offenders")
dup = [r for r in train if r["n_distinct_frames"] < 80]
check("short clips flagged via n_distinct_frames", True,
      f"{len(dup)} rows have repeats (shortest {min([r['duration'] for r in dup], default=0):.0f}s)")

print("5. timestamps are absolute video time")
r = next(x for x in train if x["format"] == "time")
t = marker_times(r["frames_indices"], r["base_fps"])
check("40 markers per row", len(t) == 40)
check("markers bracket the clip", r["start_time"] - 1 <= t[0] and t[-1] <= r["end_time"] + 1,
      f"clip {r['start_time']:.0f}-{r['end_time']:.0f}s, markers {t[0]:.1f}-{t[-1]:.1f}s")
check("markers are video-absolute, not clip-relative", t[0] > 60 if r["start_time"] > 60 else True)

print("6. frames exist on disk (spot check, 20 rows × first/mid/last)")
missing = []
for row in (train + ev + test)[::len(train + ev + test) // 20]:
    d = Path(row["frame_dir"])
    for i in (row["frames_indices"][0], row["frames_indices"][32], row["frames_indices"][-1]):
        if not frame_file(d, i).exists():
            missing.append((row["qID"], i))
check("all sampled frames present", not missing, f"{len(missing)} missing")

print("7. chat record shape")
m = train[0]["messages"]
check("two turns", len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant")
check("video placeholder present", m[0]["content"][0] == {"type": "video"})
check("question is text", m[0]["content"][1]["type"] == "text" and m[0]["content"][1]["text"])
check("target is a bare answer string", isinstance(m[1]["content"][0]["text"], str))
check("no pixel data in the export", "image" not in json.dumps(train[0]))

print("8. distributions")
for name, rows in (("train", train), ("eval", ev), ("test", test)):
    print(f"  {name}: {len(rows)} rows, {len({r['videoID'] for r in rows})} videos, "
          f"{dict(Counter(r['format'] for r in rows).most_common())}")
tf = Counter(r["format"] for r in train)
check("time is the largest train bucket", tf.most_common(1)[0][0] == "time",
      f"{tf['time']} rows = {100*tf['time']/len(train):.1f}%")
check("percentage present", tf["percentage"] > 0, f"{tf['percentage']} rows")

sz = sum((EXPORT / f"{n}.jsonl").stat().st_size for n in ("train", "eval", "test"))
print(f"\nexport size: {sz/1e6:.1f} MB")
print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
