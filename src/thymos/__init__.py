"""Thymos — a homeostatic affective memory engine.

Thymos models three coupled layers that most agent memory systems keep apart:

* a forgetting-curve memory store with consolidation and reconsolidation;
* an appraisal-driven affect system with fast emotion and slow mood;
* an endocrine layer whose hormone-like variables modulate encoding,
  retrieval and mood, and which is itself driven by circadian rhythm,
  sleep debt, weather and place.

The package is dependency-free at runtime. Everything is deterministic given
a seed, so experiments are reproducible.

    from thymos import Thymos, Profile

    mind = Thymos(profile=Profile(name="Ada", location_name="Shanghai"))
    mind.perceive("项目通过了评审，我松了一口气", importance=0.8)
    for hit in mind.recall("评审"):
        print(hit.memory.content, hit.score)
"""

from .config import Profile, ThymosConfig
from .engine import Thymos
from .agent import AgentMemory, estimate_importance
from .embedding_http import EmbeddingError, OllamaEmbedder, OpenAIEmbedder
from .facts import Fact, FactStore
from .types import Appraisal, Emotion, MemoryItem, RetrievalResult

__version__ = "0.1.2"

__all__ = [
    "Thymos",
    "AgentMemory",
    "estimate_importance",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "EmbeddingError",
    "Fact",
    "FactStore",
    "Profile",
    "ThymosConfig",
    "Appraisal",
    "Emotion",
    "MemoryItem",
    "RetrievalResult",
    "__version__",
]
