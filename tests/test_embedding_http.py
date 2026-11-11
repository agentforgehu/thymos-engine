import pytest

import thymos.embedding_http as eh


def test_ollama_embed_uses_new_endpoint(monkeypatch):
    calls = []

    def fake_post(url, payload, **kwargs):
        calls.append((url, payload))
        return {"embeddings": [[1.0, 2.0, 3.0]]}

    monkeypatch.setattr(eh, "_post_json", fake_post)
    embedder = eh.OllamaEmbedder(base_url="http://localhost:11434", model="bge-m3")
    assert embedder("你好") == [1.0, 2.0, 3.0]
    assert calls[-1][0].endswith("/api/embed")
    assert calls[-1][1]["input"] == ["你好"]


def test_ollama_falls_back_to_legacy_endpoint(monkeypatch):
    calls = []

    def fake_post(url, payload, **kwargs):
        calls.append(url)
        if url.endswith("/api/embed"):
            raise eh.EmbeddingHTTPError(404, url, "not found")
        return {"embedding": [4.0, 5.0]}

    monkeypatch.setattr(eh, "_post_json", fake_post)
    embedder = eh.OllamaEmbedder(base_url="http://localhost:11434", model="old")
    assert embedder("hello") == [4.0, 5.0]
    assert calls[-1].endswith("/api/embeddings")


def test_openai_embed_parses_data(monkeypatch):
    def fake_post(url, payload, **kwargs):
        assert url.endswith("/embeddings")
        assert kwargs["headers"]["Authorization"] == "Bearer test"
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(eh, "_post_json", fake_post)
    embedder = eh.OpenAIEmbedder(api_key="test", model="text-embedding-3-small")
    assert embedder("hello") == [0.1, 0.2]


def test_openai_embed_requires_key():
    with pytest.raises(eh.EmbeddingError):
        eh.OpenAIEmbedder(api_key="")


def test_embedding_rejects_non_numeric_vector(monkeypatch):
    monkeypatch.setattr(eh, "_post_json", lambda *a, **k: {"embeddings": [["x"]]})
    embedder = eh.OllamaEmbedder()
    with pytest.raises(eh.EmbeddingError):
        embedder("hello")
