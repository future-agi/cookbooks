"""The Pokedex agent: a LangGraph ReAct agent with one tool — Qdrant search.

The agent decides when to search, can search more than once for compound questions, and
cites its sources. A strict grounding prompt stops it from answering from the model's own
memory: Pokemon facts change across game generations, so only retrieved documents count.

Every workshop fix changes ONE thing here: how `retrieve()` searches. The retrieval mode
goes `minilm` (baseline) → `bge` (fix #2, stronger model) → `hybrid` (fix #3, dense +
sparse + rerank), plus a `current_only` filter that closes the cold open. The mode lives
in a small state file, so the notebook flips it and the running Streamlit app picks it up
on the next question.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
# One generator, any provider: GENERATOR_MODEL is a "provider:model" string, so whoever
# runs the workshop uses the API key they have. See helpers/config.py.
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
# traceAI-langchain requires langchain<0.4, so the agent uses langgraph's
# create_react_agent; switch to langchain.agents.create_agent when Future AGI
# supports langchain 1.x.
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient, models

from helpers import config, embeddings

load_dotenv()

# Future AGI tracing. Two lines: register a project, instrument LangChain. Because it
# runs at import, every process that uses the agent is traced — the notebook, the
# Streamlit app, and the rehearsal scripts. traceAI auto-instruments LangGraph, so each
# Qdrant search shows up as a retriever span with no manual span code.
if os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY"):
    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_langchain import LangChainInstrumentor

    # Project-level metadata: one label per run (stage/iteration), attached to every
    # span this process exports — set FI_STAGE_TAG before importing the agent.
    trace_provider = register(project_type=ProjectType.OBSERVE,
                              project_name=os.getenv("FI_PROJECT_NAME", "pokedex-rag"),
                              metadata={"stage": os.getenv("FI_STAGE_TAG", "adhoc")})
    LangChainInstrumentor().instrument(tracer_provider=trace_provider)
    from fi_instrumentation import FITracer
    from fi_instrumentation.fi_types import FiSpanKindValues

    tracer = FITracer(trace_provider.get_tracer(__name__))
else:
    tracer = None
    if os.getenv("FI_API_KEY") or os.getenv("FI_SECRET_KEY"):
        print("Future AGI tracing OFF: set both FI_API_KEY and FI_SECRET_KEY in .env")

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                      timeout=60)

GROUNDING_PROMPT = (
    "You are a Pokedex assistant. Answer ONLY using the documents returned by the "
    "search_pokedex tool. The Pokemon games change across generations, so your own "
    "memory is NOT reliable — treat the retrieved documents as the single source of "
    "truth. Search with the user's own wording. NEVER put a Pokemon name in a search "
    "unless it appears in the user's question or in documents you already retrieved — "
    "do not guess a name from memory and search to confirm it. If the retrieved "
    "documents do not clearly contain the answer, say you don't know. Cite the name "
    "in brackets for every fact you state, e.g. [magnemite]."
)

# Cross-process retrieval switch. The notebook flips it with set_retrieval(); the Streamlit
# app reads it per question. File-based so the two processes agree; ingest.py resets it.
# ponytail: one flag file for a single-user demo; use a real store if it ever goes multi-user.
_STATE_FILE = config.STATE_FILE


def _read_state() -> dict:
    if _STATE_FILE.exists():
        return {**config.DEFAULT_STATE, **json.loads(_STATE_FILE.read_text())}
    return dict(config.DEFAULT_STATE)


def retrieval_state() -> dict:
    """Current retrieval switch — the app shows it so the audience can track each flip."""
    return _read_state()


MODES = ("minilm", "bge", "hybrid")


def set_retrieval(*, mode: str | None = None, current_only: bool | None = None) -> None:
    """Flip the agent's retrieval behavior (persisted for the app). Called by the notebook."""
    state = _read_state()
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        state["mode"] = mode
    if current_only is not None:
        state["current_only"] = current_only
    _STATE_FILE.write_text(json.dumps(state))


# Exact search instead of approximate: it costs nothing at this collection size, and the
# live numbers reproduce exactly from rehearsal to show.
EXACT = models.SearchParams(exact=True)

# The retrieval panel reads the chunks retrieved during a run. Single-user demo.
_last_retrieval: list[dict] = []
# The exact strings handed to the model, one per tool call — multi-hop questions
# retrieve more than once, and the trace attribute must cover every hop.
_model_context: list[str] = []


def reset_retrieval() -> None:
    _last_retrieval.clear()
    _model_context.clear()


def get_retrieval() -> list[dict]:
    return list(_last_retrieval)


def get_model_context() -> str:
    """Everything the model read this question, across all retrieval hops."""
    return "\n".join(_model_context) or "No documents found."


def retrieve(query: str, *, mode: str | None = None, current_only: bool | None = None,
             limit: int = config.TOP_K) -> list[dict]:
    """Retrieve from Qdrant. `mode` selects the pipeline each workshop fix upgrades to."""
    state = _read_state()
    mode = mode or state["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    current_only = state["current_only"] if current_only is None else current_only

    # Cold-open close: keep only current documents (the payload-filter fix).
    query_filter = None
    if current_only:
        query_filter = models.Filter(must=[
            models.FieldCondition(key="is_current", match=models.MatchValue(value=True))
        ])

    if mode == "hybrid":
        # Fix #3, one query_points call: dense + sparse candidates, fused with RRF,
        # then reranked by ColBERT. The dense arm is bge, so this builds on fix #2.
        dense_query = embeddings.dense([query], config.MODEL_DENSE_STRONG, is_query=True)[0]
        indices, values = embeddings.sparse([query], is_query=True)[0]
        colbert_query = embeddings.colbert([query], is_query=True)[0]

        candidates = models.Prefetch(
            prefetch=[
                models.Prefetch(query=dense_query, using=config.DENSE_STRONG,
                                limit=20, filter=query_filter, params=EXACT),
                models.Prefetch(query=models.SparseVector(indices=indices, values=values),
                                using=config.SPARSE, limit=20, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
        )
        hits = client.query_points(
            collection_name=config.COLLECTION,
            prefetch=[candidates],
            query=colbert_query,
            using=config.COLBERT,
            limit=limit,
            with_payload=True,
        ).points
    else:
        # Baseline and fix #2: one dense query against the small (minilm) or large (bge)
        # named vector. Fix #2 is literally this `using` switch.
        model = config.MODEL_DENSE_STRONG if mode == "bge" else config.MODEL_DENSE_WEAK
        using = config.DENSE_STRONG if mode == "bge" else config.DENSE_WEAK
        hits = client.query_points(
            collection_name=config.COLLECTION,
            query=embeddings.dense([query], model, is_query=True)[0],
            using=using,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=EXACT,
        ).points

    chunks = [
        {
            "rank": rank, "score": hit.score,
            "doc_id": hit.payload["doc_id"], "name": hit.payload["name"],
            "generation": hit.payload["generation"], "doc_type": hit.payload["doc_type"],
            "sprite_url": hit.payload["sprite_url"], "text": hit.payload["text"],
        }
        for rank, hit in enumerate(hits, start=1)
    ]
    _last_retrieval.extend(chunks)
    return chunks


def _format_for_model(chunks: list[dict]) -> str:
    # The model sees name + text only, never the generation tag. If it saw the tag it
    # could tell stale documents from current ones on its own, and the cold-open failure
    # would never happen. The app's retrieval panel still shows generation badges.
    if not chunks:
        return "No documents found."
    return "\n".join(f"[{c['name']}] {c['text']}" for c in chunks)


@tool
def search_pokedex(query: str) -> str:
    """Search the Pokedex for documents relevant to the query. Returns cited chunks."""
    # Fetch PANEL_K so the panel can show what sits just below the cutoff; the model
    # only ever reads the top TOP_K.
    chunks = retrieve(query, limit=config.PANEL_K)
    formatted = _format_for_model(chunks[: config.TOP_K])
    _model_context.append(formatted)
    return formatted


llm = init_chat_model(config.GENERATOR_MODEL, temperature=0)
agent = create_react_agent(llm, [search_pokedex], prompt=GROUNDING_PROMPT)

# On a question the corpus cannot answer, the agent keeps rewording the query and searching
# again. LangGraph's default ceiling of 25 steps then raises GraphRecursionError, which would
# crash the notebook or the app mid-question. Cap it lower and answer honestly instead: a
# search that keeps coming back empty IS the "I don't know", and the trace shows the attempts.
RECURSION_LIMIT = 12
NO_ANSWER = "I don't know — I could not find that in the Pokedex."


def _invoke(messages: list[dict]) -> str:
    try:
        result = agent.invoke({"messages": messages},
                              config={"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:
        return NO_ANSWER
    return result["messages"][-1].content


def ask(question: str, history: list[dict] | None = None,
        expected_answer: str | None = None) -> tuple[str, list[dict]]:
    """Run the agent on a question, with prior chat turns so follow-ups resolve
    ("is it weak to Bug?"). `history` is [{"role": "user"|"assistant", "content": ...}].
    `expected_answer` (golden runs only) is attached to the root span so platform evals
    can judge the answer against ground truth. Returns (answer, retrieved chunks)."""
    reset_retrieval()
    messages = list(history or [])[-8:] + [{"role": "user", "content": question}]
    if tracer is None:
        return _invoke(messages), get_retrieval()
    # Root span per question: clean input/output on the trace's parent node, with the
    # auto-instrumented LangGraph tree (LLM + retriever spans) nested underneath.
    with tracer.start_as_current_span("Pokedex Agent",
                                      fi_span_kind=FiSpanKindValues.AGENT) as span:
        span.set_input(question)
        answer = _invoke(messages)
        span.set_output(answer)
        # The exact context the model read across ALL retrieval hops, as one string —
        # evals map to this directly; a multi-hop answer may cite any hop's chunks.
        chunks = get_retrieval()
        span.set_attribute("retrieved_context", get_model_context())
        if expected_answer is not None:
            span.set_attribute("expected_answer", expected_answer)
    return answer, chunks


if __name__ == "__main__":
    # ponytail: smoke-check the agent answers a working query and cites a source.
    ans, chunks = ask("Which Pokemon puts its prey to sleep and then eats their dreams?")
    print("ANSWER:", ans)
    print("RETRIEVED:", [(c["rank"], c["doc_id"], round(c["score"], 3)) for c in chunks])
