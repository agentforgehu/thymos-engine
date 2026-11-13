"""Thymos — a homeostatic affective memory control plane.

Thymos models three coupled layers that most agent memory systems keep apart:

* a forgetting-curve memory store with consolidation and reconsolidation;
* an appraisal-driven affect system with fast emotion and slow mood;
* an endocrine layer whose hormone-like variables modulate encoding,
  retrieval and mood, and which is itself driven by circadian rhythm,
  sleep debt, weather and place.

On top of that core, ``MemoryController`` routes each query to facts, episodes,
graph neighbours or schema summaries, gates whether affect may influence
retrieval, and returns an audit trace under a hard context budget.

The package is dependency-free at runtime. Everything is deterministic given
a seed, so experiments are reproducible.

    from thymos import Thymos, Profile

    mind = Thymos(profile=Profile(name="Ada", location_name="Shanghai"))
    mind.perceive("项目通过了评审，我松了一口气", importance=0.8)
    for hit in mind.recall("评审"):
        print(hit.memory.content, hit.score)
"""

from .config import Profile, ThymosConfig
from .backends import MemoryBackend
from .appraisal import explain_appraisal
from .control import AuditTrace, ContextPlan, MemoryController, OperatorReport
from .engine import Thymos
from .agent import AgentMemory, estimate_importance
from .llm_appraisal import LLMAppraiser
from .embedding_http import EmbeddingError, LlamaCppEmbedder, OllamaEmbedder, OpenAIEmbedder
from .facts import Fact, FactStore
from .policy import AffectDecision, AffectPolicy, QueryIntent, classify_intent
from .safety import redact_pii, sanitize_memory_text
from .sqlite_store import SQLiteMemoryStore
from .tokens import estimate_tokens, trim_to_tokens
from .types import Appraisal, Emotion, MemoryItem, RetrievalResult

__version__ = "0.3.5"

__all__ = [
    "Thymos",
    "AgentMemory",
    "MemoryController",
    "MemoryBackend",
    "ContextPlan",
    "AuditTrace",
    "OperatorReport",
    "QueryIntent",
    "AffectDecision",
    "AffectPolicy",
    "classify_intent",
    "estimate_importance",
    "LLMAppraiser",
    "explain_appraisal",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "LlamaCppEmbedder",
    "EmbeddingError",
    "Fact",
    "FactStore",
    "SQLiteMemoryStore",
    "sanitize_memory_text",
    "redact_pii",
    "estimate_tokens",
    "trim_to_tokens",
    "Profile",
    "ThymosConfig",
    "Appraisal",
    "Emotion",
    "MemoryItem",
    "RetrievalResult",
    "__version__",
]
