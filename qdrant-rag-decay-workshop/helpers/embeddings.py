"""FastEmbed model wrappers — lazy-loaded singletons. Camera never opens this file.

Dense (MiniLM/bge-large), sparse (miniCOIL), and late-interaction (ColBERT) all run locally
via FastEmbed, so no embedding API keys. Query vs passage embedding is distinguished
because the dense models apply a query prefix.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)

from helpers import config


# bge-v1.5 models are trained with an asymmetric query instruction that FastEmbed's
# query_embed does NOT apply. Without it, queries embed like passages and retrieval
# degrades badly. We prepend it ourselves and passage-embed the result.
_QUERY_INSTRUCTION = {
    # Every bge-v1.5 size uses this same instruction; add an entry if the model changes.
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@lru_cache(maxsize=None)
def _dense(model_id: str) -> TextEmbedding:
    return TextEmbedding(model_id)


@lru_cache(maxsize=None)
def _sparse(model_id: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_id)


@lru_cache(maxsize=None)
def _colbert(model_id: str) -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(model_id)


def dense(texts: list[str], model_id: str, *, is_query: bool = False) -> list[list[float]]:
    model = _dense(model_id)
    if is_query:
        instruction = _QUERY_INSTRUCTION.get(model_id)
        if instruction:  # bge: passage-embed the instructed query
            gen = model.embed([instruction + t for t in texts])
        else:
            gen = model.query_embed(texts)
    else:
        gen = model.embed(texts)
    return [v.tolist() for v in gen]


def sparse(texts: list[str], *, is_query: bool = False):
    """Return list of (indices, values) tuples for miniCOIL sparse vectors."""
    model = _sparse(config.MODEL_SPARSE)
    gen = model.query_embed(texts) if is_query else model.embed(texts)
    return [(v.indices.tolist(), v.values.tolist()) for v in gen]


def colbert(texts: list[str], *, is_query: bool = False) -> list[list[list[float]]]:
    """Return per-text token matrices (multivectors) for ColBERT MAX_SIM rerank."""
    model = _colbert(config.MODEL_COLBERT)
    gen = model.query_embed(texts) if is_query else model.embed(texts)
    return [m.tolist() for m in gen]


def warmup() -> None:
    """Load every model once so the first on-stage query pays no model-load latency."""
    dense(["warmup"], config.MODEL_DENSE_WEAK)
    dense(["warmup"], config.MODEL_DENSE_STRONG)
    sparse(["warmup"])
    colbert(["warmup"])
