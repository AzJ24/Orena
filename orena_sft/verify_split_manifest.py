"""Checks a regenerated export against orena_sft/split_manifest.json.

The JSONL exports are gitignored, so after rebuilding them the only way to know
the splits still match earlier runs is to compare the video->split assignment.
A drifted split silently makes new numbers incomparable to old ones.

    .venv/bin/python orena_sft/verify_split_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent


def main() -> int:
    manifest = json.loads((SFT_DIR / "split_manifest.json").read_text())
    export = SFT_DIR / "sft_export" / "combined"
    problems = []

    for split, ref in manifest["splits"].items():
        path = export / f"{split}.jsonl"
        if not path.exists():
            problems.append(f"{split}: {path} missing")
            continue
        seen = {json.loads(line)["videoID"] for line in path.open()}
        expected = set(ref["videos"])
        missing, extra = expected - seen, seen - expected
        if missing or extra:
            problems.append(
                f"{split}: {len(missing)} missing, {len(extra)} unexpected videos"
                + (f"\n    missing: {sorted(missing)[:5]}" if missing else "")
                + (f"\n    extra:   {sorted(extra)[:5]}" if extra else "")
            )
        else:
            print(f"  {split:<6} OK  {len(seen)} videos")

    if problems:
        print("\nSPLIT DRIFT:", *problems, sep="\n  ")
        return 1
    print("\nAll splits match the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
