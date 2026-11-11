"""Deterministic, dependency-free text embeddings.

This is not a semantic model.  It is a hashed bag-of-ngrams projection that
gives stable, reproducible similarity for the reference implementation and
for tests.  Production deployments can plug in a real embedding function via
`Thymos(embedder=...)`; the engine only requires a function str -> list[float].
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable

_LATIN = re.compile(r"[a-z0-9_']+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

Embedder = Callable[[str], list[float]]


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = _LATIN.findall(text)
    chars = _CJK.findall(text)
    tokens.extend(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def embed(text: str, dim: int = 256) -> list[float]:
    if dim <= 0:
        return []
    vec = [0.0] * dim
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        # Sublinear term frequency keeps repeated words from dominating.
        vec[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [v / norm for v in vec]
    return vec


def make_embedder(dim: int = 256) -> Embedder:
    def _embed(text: str) -> list[float]:
        return embed(text, dim)

    return _embed
