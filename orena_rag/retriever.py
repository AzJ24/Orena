"""Offline anatomy retriever for the FOCUS eval / container.

Loads the pre-built RAG index (passages + embeddings) and the sentence embedder,
and returns procedure-specific anatomy text keyed on procedure_type. Semantic
search with query expansion, a morphological fallback, and abstention (returns ""
when nothing is close enough -- injecting nothing beats injecting wrong anatomy).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_SUFFIX_RE = re.compile(r"(ectomy|ostomy|otomy|oscopy|plasty|pexy|rrhaphy)\b", re.I)
_STRIP = re.compile(r"\b(laparoscopic|robotic|open|total|partial|radical|elective)\b", re.I)


class AnatomyRetriever:
    def __init__(self, index_dir, embed_model="all-MiniLM-L6-v2",
                 min_score=0.58, query_suffix=" surgical anatomy", top_k=4, max_chars=600,
                 verbose=True):
        from sentence_transformers import SentenceTransformer
        index_dir = Path(index_dir)
        self.passages = [json.loads(l) for l in open(index_dir / "passages.jsonl")]
        self.emb = np.load(index_dir / "embeddings.npy")
        self.embedder = SentenceTransformer(embed_model)
        self.min_score, self.qs = min_score, query_suffix
        self.top_k, self.max_chars = top_k, max_chars
        self.verbose = verbose
        self._cache: dict[str, str] = {}

    def _search(self, query, k):
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
        sims = self.emb @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    @staticmethod
    def _root(procedure_type):
        p = _STRIP.sub("", procedure_type).strip()
        m = _SUFFIX_RE.search(p)
        return p[:m.start()].strip() if m else p

    def facts(self, procedure_type: str) -> str:
        if procedure_type in self._cache:
            return self._cache[procedure_type]
        hits = self._search(procedure_type + self.qs, self.top_k)
        if not hits or hits[0][1] < self.min_score:
            root = self._root(procedure_type)
            if root and root.lower() != procedure_type.lower():
                hits = self._search(root + self.qs, self.top_k)
        if not hits or hits[0][1] < self.min_score:
            self._cache[procedure_type] = ""
            if self.verbose:
                top = f"{hits[0][1]:.3f}" if hits else "n/a"
                print(f"[RAG] {procedure_type!r}: ABSTAIN (top score {top} < {self.min_score})", flush=True)
            return ""
        text = " ".join(self.passages[i]["text"] for i, _ in hits)[:self.max_chars]
        self._cache[procedure_type] = text
        if self.verbose:
            titles = [self.passages[i]["title"] for i, _ in hits]
            print(f"\n[RAG] {procedure_type!r} (top score {hits[0][1]:.3f}) -> {titles}\n"
                  f"      {text[:400]}{'...' if len(text) > 400 else ''}\n", flush=True)
        return text
