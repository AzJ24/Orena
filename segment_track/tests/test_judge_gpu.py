"""Reproduce the scoring crash: does the focus judge load and run on this GPU?

Job 12955 merged 4,000 predictions, then died inside the judge with
"'NoneType' object is not a mapping" from Triton's autotuner. The judge is
Qwen3.5-4B -- the same hybrid architecture, but different head dims from the
27B, so it autotunes different kernel shapes. This isolates that in ~2 min
instead of re-running a 40-minute merge to find out.
"""
import sys, traceback
import torch
print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))

from focus.evaluation.judges import TransformersJudge, DEFAULT_JUDGE_MODEL
from focus.data.data_models import Request

print("judge model:", DEFAULT_JUDGE_MODEL)
try:
    judge = TransformersJudge(device="cuda")
    req = Request(qID="probe", videoID="probe.mp4", start_time=0.0, end_time=29.0,
                  procedure_type="laparoscopic cholecystectomy",
                  question="What instrument is used to cut tissue?")
    # Two calls: the first triggers Triton autotuning on the judge's kernel
    # shapes, which is where job 12955 died.
    print("verdict 1:", judge.judge(req, "scissors", "a pair of scissors"))
    print("verdict 2:", judge.judge(req, "scissors", "a clip applier"))
    print("\nRESULT: judge works on this GPU")
except Exception:
    traceback.print_exc()
    print("\nRESULT: judge FAILED on this GPU")
    sys.exit(1)
