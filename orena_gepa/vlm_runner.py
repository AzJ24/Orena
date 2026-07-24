"""Thin wrapper that loads the VLM once and generates answers for a batch of
(image, question) pairs under a given system prompt.

Kept separate from the GEPA adapter so the whole scoring/feedback layer stays
importable and testable without a GPU or the model weights. The generation flow
mirrors `orena_sft/base_model_eval.py` (left-padded batch, empty closed think
block, strip thinking, `extract_answer` on the visible output)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orena_sft"))
from base_model_eval import strip_thinking  # noqa: E402
from prompts import extract_answer  # noqa: E402


class VLMRunner:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", max_new_tokens: int = 128,
                 batch_size: int = 16):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto",
        )
        self.model.eval()

    @torch.no_grad()
    def _generate_chunk(self, system_prompt: str, examples: list[dict]) -> list[tuple[str, str]]:
        system_turn = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        convs = [
            system_turn + [{"role": "user", "content": [
                {"type": "image", "image": Image.open(ex["image_path"]).convert("RGB")},
                {"type": "text", "text": ex["question"]},
            ]}]
            for ex in examples
        ]
        inputs = self.processor.apply_chat_template(
            convs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
            processor_kwargs={"padding": True},
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        results = []
        for row in outputs:
            raw = self.processor.decode(row[input_len:], skip_special_tokens=True)
            _, content = strip_thinking(raw)
            results.append((extract_answer(content), raw))
        return results

    def generate(self, system_prompt: str, examples: list[dict]) -> list[tuple[str, str]]:
        """Return one (parsed_answer, raw_output) per example, batched internally."""
        out: list[tuple[str, str]] = []
        for i in range(0, len(examples), self.batch_size):
            out.extend(self._generate_chunk(system_prompt, examples[i:i + self.batch_size]))
        return out
