"""Conservative token-budget helpers.

The core project is dependency-free and does not know the caller's tokenizer.
These helpers deliberately over-estimate Chinese characters (one character is
usually one token or less) and count Latin words as 1.3 tokens.  They are meant
for hard upper bounds, not billing.
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_SPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    other = len(_SPACE.sub("", text)) - cjk - sum(len(w) for w in _WORD.findall(text))
    return max(1, cjk + int(words * 1.3) + int(max(0, other) * 0.35))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Trim from the end until the estimate fits.  Keeps a short ellipsis."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"
