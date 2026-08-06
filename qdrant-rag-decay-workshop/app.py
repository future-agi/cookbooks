"""Pokedex chat UI — the audience-facing surface where the failure is experienced.

Left: chat with the agent (multi-turn: prior turns are passed back to the agent).
Right: the retrieval panel showing exactly which chunks Qdrant returned for the last
question — sprite, doc_id, generation, rank, similarity. The panel exposes each failure
before any eval confirms it. Eval scores never appear here; they live on the Future AGI
dashboard. Theme: .streamlit/config.toml (light) + helpers/ui.py (Pokedex chrome).

    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from agent import ask, retrieval_state
from helpers import ui


@st.cache_resource
def _warm_models() -> bool:
    from helpers import embeddings

    embeddings.warmup()  # pay all model-load latency at boot, never on a live question
    return True


@st.cache_resource
def _card_index() -> dict:
    """Full documents for the click-to-unfold card: whole text by doc_id, plus each
    Pokemon's current types and stats. Served from data/corpus_cache.json (first ever
    run fetches PokeAPI, same cost ingest.py pays)."""
    from helpers.corpus import build_corpus

    docs = build_corpus()
    by_name: dict[str, dict] = {}
    for d in docs:
        if d["doc_type"] in ("types", "stats") and d["is_current"]:
            by_name.setdefault(d["name"], {})[d["doc_type"]] = d["text"]
    return {"by_id": {d["doc_id"]: d for d in docs}, "by_name": by_name}


st.set_page_config(page_title="Pokedex RAG", page_icon="🔴", layout="wide")
st.markdown(ui.POKEDEX_CSS, unsafe_allow_html=True)
st.markdown(ui.HEADER_HTML, unsafe_allow_html=True)

with st.spinner("Loading embedding models…"):
    _warm_models()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, answer, chunks}

chat_col, panel_col = st.columns([3, 2], gap="large")

with chat_col:
    # chat_input only pins to the page bottom at top level; inside a column it renders
    # in source order, so keep a container above it for the messages.
    messages = st.container()
    with messages:
        for turn in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant", avatar="🔴"):
                st.markdown(turn["answer"])

    question = st.chat_input("Ask the Pokedex…")
    if question:
        past = [m for turn in st.session_state.history for m in (
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "content": turn["answer"]},
        )]
        with messages:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🔴"), st.spinner("Searching the Pokedex…"):
                answer, chunks = ask(question, history=past)
                st.markdown(answer)
        st.session_state.history.append(
            {"question": question, "answer": answer, "chunks": chunks}
        )

with panel_col:
    st.markdown(
        f'<div class="panel-head">🔍 Retrieved chunks {ui.mode_badge_html(retrieval_state())}</div>',
        unsafe_allow_html=True,
    )
    last = st.session_state.history[-1] if st.session_state.history else None
    st.markdown(ui.retrieval_panel_html(last["chunks"] if last else [], cards=_card_index()),
                unsafe_allow_html=True)
