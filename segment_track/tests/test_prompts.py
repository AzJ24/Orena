"""Frame prompt must be unchanged; segment prompt must cover every segment format.

The frame arm is compared against the committed `prompts.py` from git, not
against a hardcoded copy -- every existing frame checkpoint was trained with that
exact string, and a one-character drift silently invalidates their eval numbers.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

SFT = Path(__file__).resolve().parents[2] / "orena_sft"
sys.path.insert(0, str(SFT))

import prompts as new  # noqa: E402

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def load_committed():
    src = subprocess.run(["git", "show", "HEAD:orena_sft/prompts.py"], cwd=SFT.parent,
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "prompts_committed.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("prompts_committed", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("1. FRAME arm is byte-identical to the committed prompt")
old = load_committed()
for style in ("structured", "direct"):
    for defs in (False, True):
        a = old.build_system_prompt(include_definitions=defs, style=style)
        b = new.build_system_prompt(include_definitions=defs, style=style)
        check(f"style={style} defs={defs}", a == b,
              "identical" if a == b else f"{len(a)} vs {len(b)} chars")
check("track='frame' explicit == default",
      new.build_system_prompt(style="direct") == new.build_system_prompt(style="direct", track="frame"))

print("2. SEGMENT arm content")
seg = new.build_system_prompt(style="direct", track="segment")
frame = new.build_system_prompt(style="direct", track="frame")
check("differs from frame prompt", seg != frame)
check("describes a clip, not one frame", "short clip" in seg and "one frame" not in seg)
check("explains the <SECONDS seconds> marker", "<SECONDS seconds>" in seg)
check("worked example of the conversion", "<1234.5 seconds>" in seg and "00:20:34" in seg)
# The illustrative value must not look like a real frame marker for these clips --
# a prompt-side literal that a parser (or the model) could mistake for data.
check("example value is obviously illustrative", "<559.8 seconds>" not in seg)
check("says timestamps are video-absolute", "START OF THE VIDEO" in seg)
check("tells it to interpolate between frames", "between two of them" in seg
      or "between their timestamps" in seg)

print("3. every segment answer_format has a rule")
for fmt, needle in [
    ("binary", "write exactly: yes"),
    ("number", "digits only"),
    ("fo_class", "class names exactly as spelled"),
    ("time", "write hh:mm:ss"),
    ("percentage", "percentage or a share"),
    ("multiple_choice", "copy exactly one of those options"),
    ("open_ended", "a short phrase"),
]:
    check(f"{fmt}", needle in seg)
check("multi-timestamp answers covered", "00:09:19, 00:12:44" in seg)

print("4. formats the segment prompt must NOT invent")
check("no REASONING line in direct mode", "REASONING" not in seg)
check("structured mode still offers one", "REASONING" in new.build_system_prompt(
    style="structured", track="segment"))
check("FO class vocabulary injected live", "Clip" in seg and "Sponge" in seg)

print("5. bad arguments rejected")
for kwargs in ({"track": "procedure"}, {"style": "freeform"}):
    try:
        new.build_system_prompt(**kwargs)
        check(f"rejects {kwargs}", False, "no error raised")
    except ValueError:
        check(f"rejects {kwargs}", True)

print(f"\n--- SEGMENT PROMPT ({len(seg)} chars) ---\n{seg}")
print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
