"""Optional HTTP embedding backends.

The core package stays zero-dependency: this module is only imported when a
host explicitly asks for a remote embedding.  Both backends use ``urllib`` and
the standard library.

Ollama (recommended for Chinese, e.g. ``bge-m3``)::

    from thymos.embedding_http import OllamaEmbedder
    embedder = OllamaEmbedder(model="bge-m3")
    mind = Thymos(embedder=embedder)

OpenAI-compatible (any ``/embeddings`` endpoint)::

    from thymos.embedding_http import OpenAIEmbedder
    embedder = OpenAIEmbedder(api_key="...", model="text-embedding-3-small")
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot be reached or parsed."""


class EmbeddingHTTPError(EmbeddingError):
    """HTTP error with a status code, so callers can fall back on 404."""

    def __init__(self, status: int, url: str, detail: str = "") -> None:
        self.status = status
        self.url = url
        self.detail = detail
        message = f"HTTP {status} from {url}"
        if detail:
            message += f": {detail[:500]}"
        super().__init__(message)


def _normalize_base(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if not value:
        raise EmbeddingError("embedding base URL 不能为空")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def _endpoint(base: str, path: str) -> str:
    base = _normalize_base(base)
    if base.endswith(path):
        return base
    return base + path


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "thymos-embedding/0.1",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise EmbeddingHTTPError(exc.code, url, detail) from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"无法连接 {url}：{exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise EmbeddingError(f"embedding 请求超时（{timeout:g}s）：{url}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EmbeddingError(f"embedding 服务返回的不是 JSON（{url}）：{raw[:300]}") from exc
    if not isinstance(data, dict):
        raise EmbeddingError(f"embedding 服务返回了意外的 JSON 类型：{type(data).__name__}")
    return data


def _floats(values: Any) -> list[float]:
    if not isinstance(values, list) or not values:
        raise EmbeddingError("embedding 响应里没有非空向量")
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"embedding 向量包含非数值：{values[:5]}") from exc


class OllamaEmbedder:
    """Ollama ``/api/embed`` with a fallback to legacy ``/api/embeddings``.

    ``bge-m3`` is a good multilingual/Chinese default.  Vectors are cached by
    text, which matters because ``recall`` embeds the same query repeatedly.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        *,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = _normalize_base(base_url)
        self.model = model
        self.timeout = timeout

    @lru_cache(maxsize=1024)
    def __call__(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            return self._embed_new(text)
        except EmbeddingHTTPError as exc:
            if exc.status != 404:
                raise
            return self._embed_legacy(text)

    def _embed_new(self, text: str) -> list[float]:
        data = _post_json(
            _endpoint(self.base_url, "/api/embed"),
            {"model": self.model, "input": [text]},
            timeout=self.timeout,
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return _floats(embeddings[0])
        raise EmbeddingError(f"Ollama /api/embed 响应缺少 embeddings：{str(data)[:300]}")

    def _embed_legacy(self, text: str) -> list[float]:
        data = _post_json(
            _endpoint(self.base_url, "/api/embeddings"),
            {"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        embedding = data.get("embedding")
        if embedding is None:
            raise EmbeddingError(f"Ollama /api/embeddings 响应缺少 embedding：{str(data)[:300]}")
        return _floats(embedding)


class OpenAIEmbedder:
    """Any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        *,
        timeout: float = 60.0,
    ) -> None:
        if not (api_key or "").strip():
            raise EmbeddingError("OpenAIEmbedder 需要 api_key")
        self.api_key = api_key.strip()
        self.base_url = _normalize_base(base_url)
        self.model = model
        self.timeout = timeout

    @lru_cache(maxsize=1024)
    def __call__(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return []
        data = _post_json(
            _endpoint(self.base_url, "/embeddings"),
            {"model": self.model, "input": text},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise EmbeddingError(f"OpenAI embedding 响应缺少 data：{str(data)[:300]}")
        first = items[0]
        if not isinstance(first, dict) or "embedding" not in first:
            raise EmbeddingError(f"OpenAI embedding 响应格式异常：{str(first)[:300]}")
        return _floats(first["embedding"])


__all__ = [
    "EmbeddingError",
    "EmbeddingHTTPError",
    "OllamaEmbedder",
    "OpenAIEmbedder",
]
