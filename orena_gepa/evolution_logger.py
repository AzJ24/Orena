"""Follow the prompt as GEPA evolves it: a tee'd run log, a per-step record of
every proposed prompt (accepted or rejected) with its score, and a final
start-vs-end summary with a diff.

Three artefacts land in the run dir:
  * run_log.txt      -- GEPA's full stdout, tee'd to disk.
  * evolution.jsonl  -- one row per candidate/proposal (machine-readable).
  * evolution.md     -- human-readable timeline: seed prompt, each accepted
                        prompt with a unified diff vs its parent, and a final
                        SEED vs BEST section.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path


class TeeLogger:
    """GEPA LoggerProtocol: prints every engine message and appends it to a file,
    so the whole iteration-by-iteration trace is on disk as well as on screen."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")

    def log(self, message: str):
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()


def _text(instr) -> str:
    """new_instructions values are usually str; tolerate a messages-list form."""
    if isinstance(instr, str):
        return instr
    if isinstance(instr, list):
        return "\n".join(p.get("text", "") for p in instr if isinstance(p, dict))
    return str(instr)


def _diff(old: str, new: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="parent", tofile="proposed", lineterm="",
    )
    return "\n".join(lines) or "(no textual change)"


class EvolutionTracker:
    """GEPA callback that records the lineage of the evolving prompt.

    Only the hooks used here are defined; the callback manager skips the rest.
    """

    def __init__(self, out_dir: Path, component: str = "system_prompt"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.component = component
        # Overwrite (not append) so re-running an arm starts a clean log.
        self.jsonl = (self.out_dir / "evolution.jsonl").open("w")
        self.md = (self.out_dir / "evolution.md").open("w")
        self.texts: dict[int, str] = {}          # candidate idx -> prompt text
        self._pending: dict[int, dict] = {}       # iteration -> {text, raw}

    # -- lifecycle -------------------------------------------------------
    def on_optimization_start(self, event):
        seed = event["seed_candidate"].get(self.component, "")
        self.texts[0] = seed
        self._emit({"kind": "seed", "idx": 0, "score": None, "prompt": seed})
        self.md.write(
            f"# Prompt evolution\n\n"
            f"train={event.get('trainset_size')}  val={event.get('valset_size')}\n\n"
            f"## Seed prompt (candidate 0)\n\n```\n{seed}\n```\n\n"
        )
        self.md.flush()
        print(f"\n[evolution] seed prompt logged ({len(seed)} chars) -> {self.out_dir}/evolution.md")

    def on_proposal_end(self, event):
        instr = event.get("new_instructions", {}).get(self.component)
        raw = (event.get("raw_lm_outputs") or {}).get(self.component, "")
        if instr is not None:
            self._pending[event["iteration"]] = {"text": _text(instr), "raw": raw}
            print(f"[evolution] iter {event['iteration']}: reflection LM proposed a new prompt")

    def on_candidate_accepted(self, event):
        it = event["iteration"]
        idx = event["new_candidate_idx"]
        text = self._pending.get(it, {}).get("text", "")
        parent = event["parent_ids"][0] if event.get("parent_ids") else 0
        self.texts[idx] = text
        parent_text = self.texts.get(parent, "")
        self._emit({"kind": "accepted", "idx": idx, "iteration": it, "parent": parent,
                    "score": event["new_score"], "prompt": text,
                    "raw_reflection": self._pending.get(it, {}).get("raw", "")})
        self.md.write(
            f"## ✅ Accepted candidate {idx}  (iter {it}, parent {parent}, "
            f"minibatch score {event['new_score']:.4f})\n\n"
            f"### diff vs parent {parent}\n```diff\n{_diff(parent_text, text)}\n```\n\n"
            f"### full prompt\n```\n{text}\n```\n\n"
        )
        self.md.flush()
        print(f"[evolution] iter {it}: ACCEPTED candidate {idx}  "
              f"minibatch score={event['new_score']:.4f} (val aggregate in final summary)")

    def on_candidate_rejected(self, event):
        it = event["iteration"]
        self._emit({"kind": "rejected", "iteration": it,
                    "old_score": event.get("old_score"), "new_score": event.get("new_score"),
                    "reason": event.get("reason"),
                    "prompt": self._pending.get(it, {}).get("text", "")})
        print(f"[evolution] iter {it}: rejected  "
              f"{event.get('old_score')} -> {event.get('new_score')} ({event.get('reason')})")

    # -- helpers ---------------------------------------------------------
    def _emit(self, row: dict):
        self.jsonl.write(json.dumps(row) + "\n")
        self.jsonl.flush()

    def close(self):
        self.jsonl.close()
        self.md.close()


def write_final_summary(result, out_dir: Path, component: str = "system_prompt"):
    """Authoritative start-vs-end summary from the GEPAResult (independent of the
    live callback), appended to evolution.md and printed."""
    out_dir = Path(out_dir)
    candidates = result.candidates
    scores = result.val_aggregate_scores
    parents = getattr(result, "parents", None)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    seed_text = candidates[0].get(component, "")
    best_text = candidates[best_idx].get(component, "")

    lineage = "\n".join(
        f"| {i} | {parents[i] if parents else '-'} | {scores[i]:.4f} |"
        for i in range(len(candidates))
    )
    body = (
        f"\n---\n\n# Final summary\n\n"
        f"Total candidates: {len(candidates)}  |  best: candidate {best_idx}  "
        f"(val {scores[best_idx]:.4f}, seed was {scores[0]:.4f}, "
        f"Δ {scores[best_idx] - scores[0]:+.4f})\n\n"
        f"## Lineage\n\n| idx | parent | val score |\n|--|--|--|\n{lineage}\n\n"
        f"## SEED (candidate 0, val {scores[0]:.4f})\n\n```\n{seed_text}\n```\n\n"
        f"## BEST (candidate {best_idx}, val {scores[best_idx]:.4f})\n\n```\n{best_text}\n```\n\n"
        f"## SEED → BEST diff\n\n```diff\n{_diff(seed_text, best_text)}\n```\n"
    )
    with (out_dir / "evolution.md").open("a") as f:
        f.write(body)
    (out_dir / "best_prompt.txt").write_text(best_text)
    (out_dir / "seed_prompt.txt").write_text(seed_text)

    print(f"\n{'=' * 70}\nEVOLUTION: seed val={scores[0]:.4f} -> best val={scores[best_idx]:.4f} "
          f"(Δ {scores[best_idx] - scores[0]:+.4f}), candidate {best_idx} of {len(candidates)}")
    print(f"full timeline -> {out_dir}/evolution.md")
    print(f"best prompt   -> {out_dir}/best_prompt.txt\n{'=' * 70}")
    return best_idx
