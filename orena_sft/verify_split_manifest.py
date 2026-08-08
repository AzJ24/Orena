"""Checks a regenerated export against a split manifest.

The JSONL exports are gitignored, so after rebuilding them the only way to know
the splits still match earlier runs is to compare against the pinned manifest.
A drifted split silently makes new numbers incomparable to old ones.

    .venv/bin/python orena_sft/verify_split_manifest.py                # v2
    .venv/bin/python orena_sft/verify_split_manifest.py \
        --manifest orena_sft/split_manifest.json \
        --export orena_sft/sft_export/combined                         # legacy

Manifests carrying `qid_sha256` are checked row-exactly; older ones only pin
the video assignment, which is not sufficient once eval rows are subsampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent


def qid_digest(rows: list[dict]) -> str:
    keys = sorted(f"{r['source_dataset']}:{r['qID']}" for r in rows)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=SFT_DIR / "split_manifest_v2.json")
    ap.add_argument("--export", type=Path, default=SFT_DIR / "sft_export" / "combined_v2")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    print(f"manifest: {args.manifest}\nexport:   {args.export}")
    print("params:  ", json.dumps(manifest.get("params", {}), sort_keys=True))
    problems = []
    rows_by_split: dict[str, list[dict]] = {}

    for split, ref in manifest["splits"].items():
        path = args.export / f"{split}.jsonl"
        if not path.exists():
            problems.append(f"{split}: {path} missing")
            continue
        rows = [json.loads(line) for line in path.open()]
        rows_by_split[split] = rows

        seen = {r["videoID"] for r in rows}
        expected = set(ref["videos"])
        missing, extra = expected - seen, seen - expected
        if missing or extra:
            problems.append(
                f"{split}: {len(missing)} missing, {len(extra)} unexpected videos"
                + (f"\n    missing: {sorted(missing)[:5]}" if missing else "")
                + (f"\n    extra:   {sorted(extra)[:5]}" if extra else "")
            )
            continue

        if "n_rows" in ref and len(rows) != ref["n_rows"]:
            problems.append(f"{split}: {len(rows)} rows, manifest says {ref['n_rows']}")
            continue

        if "qid_sha256" in ref:
            got = qid_digest(rows)
            if got != ref["qid_sha256"]:
                problems.append(f"{split}: qID digest {got[:12]} != {ref['qid_sha256'][:12]}")
                continue
            print(f"  {split:<6} OK  {len(seen):3d} videos, {len(rows):5d} rows, qIDs match")
        else:
            print(f"  {split:<6} OK  {len(seen):3d} videos (video-level only, no qID digest)")

    # Leakage is a property of the export, not of the manifest, so check it here
    # rather than trusting that whoever built the manifest checked it.
    if {"train", "eval"} <= rows_by_split.keys():
        tr, ev = rows_by_split["train"], rows_by_split["eval"]
        vid = {r["videoID"] for r in tr} & {r["videoID"] for r in ev}
        qid = ({(r["source_dataset"], r["qID"]) for r in tr}
               & {(r["source_dataset"], r["qID"]) for r in ev})
        if vid:
            problems.append(f"train/eval share {len(vid)} videos")
        if qid:
            problems.append(f"train/eval share {len(qid)} qIDs")
        if not vid and not qid:
            print("  leakage OK  train/eval share no videos and no qIDs")

    if problems:
        print("\nSPLIT DRIFT:", *problems, sep="\n  ")
        return 1
    print("\nAll splits match the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
