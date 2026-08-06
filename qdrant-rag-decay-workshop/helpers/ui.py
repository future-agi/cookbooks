"""Streamlit UI plumbing — Pokedex chrome + retrieval-panel rendering. Camera never opens this.

The light theme itself lives in .streamlit/config.toml; this CSS only styles the custom
Pokedex elements, so it never fights Streamlit's own widgets (the old global color
override made typed text invisible in the chat input).

All HTML is emitted WITHOUT newlines or leading indentation: st.markdown parses markdown
even with unsafe_allow_html, and any line indented 4+ spaces becomes a code block, which
is exactly how the panel broke.

The retrieval panel makes each failure visible BEFORE any score — duplicates repeating a
doc_id, a stale generation badge on the winning chunk, the right chunk ranked low.
Hard rule: no eval scores here, ever. Similarity score is a retrieval signal, not a judgment.
"""

from __future__ import annotations

import html

from helpers import config

POKEDEX_CSS = """
<style>
  .dex-top { background: linear-gradient(#d92626, #b30000); border: 3px solid #7a0000;
    border-radius: 14px; box-shadow: 0 5px 0 #7a0000; padding: 12px 20px;
    display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .dex-lens { width: 46px; height: 46px; border-radius: 50%; flex: none;
    background: radial-gradient(circle at 35% 35%, #cfe9ff, #2f7fd1 55%, #10386b);
    border: 3px solid #f2f2f2; box-shadow: inset 0 0 8px rgba(0,0,0,.45); }
  .dex-led { width: 13px; height: 13px; border-radius: 50%; flex: none;
    border: 2px solid rgba(0,0,0,.3); }
  .dex-led.r { background: #ff5a5a; } .dex-led.y { background: #ffd83d; }
  .dex-led.g { background: #4cd964; }
  .dex-title { color: #ffffff; font-weight: 900; font-size: 27px; letter-spacing: 1.5px;
    text-shadow: 0 2px 0 rgba(0,0,0,.35); margin-left: 6px; }
  .dex-title span { color: #ffde00; font-size: 16px; letter-spacing: .5px; margin-left: 10px; }

  .panel-head { font-size: 20px; font-weight: 800; color: #1a1a2e; margin: 4px 0 10px; }
  .mode-badge { background: #1a1a2e; color: #ffde00; border-radius: 6px; padding: 2px 10px;
    font-size: 14px; font-weight: 700; font-family: ui-monospace, monospace;
    vertical-align: middle; margin-left: 8px; }

  .dex-screen { background: #dde8d8; border: 3px solid #2a3a5e; border-radius: 12px;
    padding: 10px; box-shadow: inset 0 2px 6px rgba(0,0,0,.15); }
  .chunk-card { background: #ffffff; border: 2px solid #2a3a5e; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 8px; display: grid;
    grid-template-columns: 64px 1fr; gap: 12px; align-items: center; }
  .chunk-card.dup { border-color: #e08a00; background: #fff6e6; }
  .sprite { width: 64px; height: 64px; image-rendering: pixelated; }
  .chunk-meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 3px; }
  .rank-badge { background: #2a3a5e; color: #ffffff; font-weight: 800; border-radius: 6px;
    padding: 1px 8px; font-size: 14px; }
  .name-badge { font-weight: 800; font-size: 17px; text-transform: capitalize; color: #1a1a2e; }
  .gen-badge { background: #3b6ea5; color: #ffffff; border-radius: 6px; padding: 1px 8px;
    font-size: 13px; font-weight: 700; }
  .type-badge { background: #eef1f6; border: 1px solid #c7d0e0; color: #37415c;
    border-radius: 6px; padding: 1px 8px; font-size: 13px; }
  .dup-badge { background: #e08a00; color: #ffffff; border-radius: 6px; padding: 1px 8px;
    font-size: 13px; font-weight: 700; }
  .cutoff { text-align: center; font-size: 13px; font-weight: 700; color: #7a0000;
    border-top: 2px dashed #b30000; margin: 4px 2px 10px; padding-top: 4px; }
  .docid { font-family: ui-monospace, monospace; font-size: 13px; color: #55607a; }
  .chunk-text { font-size: 15px; color: #1a1a2e; margin: 3px 0 5px; }
  .score-row { display: flex; align-items: center; gap: 8px; }
  .score-track { flex: 1; height: 10px; background: #e6e9f0; border-radius: 5px; overflow: hidden; }
  .score-fill { height: 100%; background: #30a14e; }
  .score-num { font-variant-numeric: tabular-nums; font-size: 13px; font-weight: 700;
    color: #1a1a2e; }

  details.chunk-wrap > summary { list-style: none; cursor: pointer; }
  details.chunk-wrap > summary::-webkit-details-marker { display: none; }
  details.chunk-wrap > summary .chunk-card:hover { box-shadow: 0 2px 6px rgba(42,58,94,.25); }
  .full-card { background: #f7fafc; border: 2px solid #2a3a5e; border-radius: 10px;
    margin: -4px 0 10px; padding: 12px 14px; display: grid;
    grid-template-columns: 96px 1fr; gap: 14px; align-items: start; }
  .sprite-lg { width: 96px; height: 96px; image-rendering: pixelated; }
  .full-line { font-size: 14px; color: #1a1a2e; margin: 4px 0; }
  .full-label { font-weight: 800; color: #2a3a5e; margin-right: 6px; }
</style>
"""

HEADER_HTML = (
    '<div class="dex-top"><div class="dex-lens"></div>'
    '<div class="dex-led r"></div><div class="dex-led y"></div><div class="dex-led g"></div>'
    '<div class="dex-title">POKéDEX <span>Agentic RAG · powered by Qdrant</span></div></div>'
)


def _card(chunk: dict, is_dup: bool, cards: dict | None = None) -> str:
    dup_cls = " dup" if is_dup else ""
    dup_badge = '<span class="dup-badge">\U0001f501 duplicate</span>' if is_dup else ""
    sprite = chunk["sprite_url"] or ""
    img = f'<img class="sprite" src="{html.escape(sprite)}"/>' if sprite else '<div class="sprite"></div>'
    pct = max(2, min(100, round(chunk["score"] * 100)))
    card = (
        f'<div class="chunk-card{dup_cls}">{img}<div>'
        f'<div class="chunk-meta"><span class="rank-badge">#{chunk["rank"]}</span>'
        f'<span class="name-badge">{html.escape(chunk["name"])}</span>'
        f'<span class="gen-badge">Gen {chunk["generation"]}</span>'
        f'<span class="type-badge">{html.escape(chunk["doc_type"].replace("_", " "))}</span>{dup_badge}</div>'
        f'<div class="docid">{html.escape(chunk["doc_id"])}</div>'
        f'<div class="chunk-text">{html.escape(chunk["text"])}</div>'
        f'<div class="score-row"><div class="score-track">'
        f'<div class="score-fill" style="width:{pct}%"></div></div>'
        f'<span class="score-num">{chunk["score"]:.3f}</span></div>'
        f'</div></div>'
    )
    if cards is None:
        return card
    # Click to unfold the full card: whole source document (chunks are truncated) plus
    # the Pokemon's current types and stats. Native <details>, no JS survives st.markdown.
    return f'<details class="chunk-wrap"><summary>{card}</summary>{_full_card(chunk, cards)}</details>'


def _full_card(chunk: dict, cards: dict) -> str:
    doc = cards["by_id"].get(chunk["doc_id"])
    full_text = doc["text"] if doc else chunk["text"]
    sprite = chunk["sprite_url"] or ""
    img = f'<img class="sprite-lg" src="{html.escape(sprite)}"/>' if sprite else '<div class="sprite-lg"></div>'
    # Everything in the unfolded card beyond the retrieved fragment is reference data
    # from the local corpus cache — labeled so it never reads as model context.
    lines = [f'<div class="full-line"><span class="full-label">Full source document (agent saw the fragment above)</span>'
             f'{html.escape(full_text)}</div>']
    for label, key in (("Types (reference)", "types"), ("Stats (reference)", "stats")):
        extra = cards["by_name"].get(chunk["name"], {}).get(key)
        if extra:
            lines.append(f'<div class="full-line"><span class="full-label">{label}</span>{html.escape(extra)}</div>')
    return f'<div class="full-card">{img}<div>{"".join(lines)}</div></div>'


def mode_badge_html(state: dict) -> str:
    """Retrieval-mode badge for the panel header — state, never a judgment."""
    label = state["mode"] + (" · current-only" if state.get("current_only") else "")
    return f'<span class="mode-badge">{html.escape(label)}</span>'


def retrieval_panel_html(chunks: list[dict], cards: dict | None = None) -> str:
    """Render retrieved chunks as cards on the Pokedex screen, flagging repeated doc_ids.

    The panel shows PANEL_K chunks per search but the agent only reads the top TOP_K,
    so a cutoff line marks where the agent stops — what sits below it (the current
    Steel chart at the cold open) is visible to the audience and invisible to the agent.
    """
    if not chunks:
        return '<div class="dex-screen"><div class="chunk-text">No documents retrieved yet — ask a question.</div></div>'
    # Duplicate = same fragment already shown, keyed on (doc_id, text) — the same key the
    # notebook's dup_rate and verify_arc.py use, so the panel and the printed rate agree.
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for c in chunks:
        if c["rank"] == config.TOP_K + 1:  # ranks restart at 1 for each agent search
            out.append(f'<div class="cutoff">▲ top {config.TOP_K} — all the agent reads</div>')
        is_dup = (c["doc_id"], c["text"]) in seen
        seen.add((c["doc_id"], c["text"]))
        out.append(_card(c, is_dup, cards))
    return '<div class="dex-screen">' + "".join(out) + "</div>"
