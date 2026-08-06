"""Rehearsal gate: run the whole workshop arc and prove each fix moves its metric.

Measures the golden set at every stage of the arc — baseline -> dedup -> strong embedding
-> hybrid+rerank -> is_current filter — and prints the metric that each fix is supposed
to move. DESTRUCTIVE (dedup deletes points, the filter adds an index); revert with
`snapshot.py restore` afterwards. It refuses to run without a snapshot on the cluster.

    uv run python verify_arc.py                  # full arc (destructive)
    uv run python verify_arc.py --baseline-only  # non-destructive: flaws show red?

These are retrieval proxies (computed from gold labels); the scored evals run on the
Future AGI platform. If a number here doesn't move, the beat won't move on the dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

import agent
from helpers import config
from helpers.dedup import find_duplicate_ids

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_dataset.jsonl"
K = config.TOP_K


def load(group=None):
    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    return [r for r in rows if group is None or r["exercises"] == group]


def _hits(query, **kw):
    return agent.retrieve(query, limit=30, **kw)


def _ranked_docs(query, **kw):
    """Unique doc_ids in rank order (a doc that appears as several chunks counts once)."""
    seen, out = set(), []
    for c in _hits(query, **kw):
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            out.append(c["doc_id"])
    return out


def recall_at_k(rows, **kw):
    hit = 0
    for r in rows:
        gold = set(r["gold_doc_ids"])
        top = _ranked_docs(r["query"], **kw)[:K]
        hit += int(any(d in gold for d in top))
    return hit / len(rows)


def ndcg_mrr(rows, **kw):
    ndcgs, rrs = [], []
    for r in rows:
        gold = set(r["gold_doc_ids"])
        ranked = _ranked_docs(r["query"], **kw)[:K]
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), K)))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        rr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in gold), 0.0)
        rrs.append(rr)
    return sum(ndcgs) / len(ndcgs), sum(rrs) / len(rrs)


def dup_rate(rows, **kw):
    """Fraction of top-k slots that are redundant (repeat a (doc_id, text))."""
    rates = []
    for r in rows:
        chunks = _hits(r["query"], **kw)[:K]
        keys = [(c["doc_id"], c["text"]) for c in chunks]
        rates.append(1 - len(set(keys)) / len(keys) if keys else 0.0)
    return sum(rates) / len(rates)


def cold_open_correct(**kw):
    """Is the current Steel chart ranked above every stale one?"""
    r = load("cold_open")[0]
    top_steel = next((c["doc_id"] for c in _hits(r["query"], **kw) if c["name"] == "steel"),
                     None)
    return top_steel == r["gold_doc_ids"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true",
                    help="stop after the baseline read (non-destructive pre-show check)")
    args = ap.parse_args()

    load_dotenv()
    # Reset the switch first: retrieve() falls back to file state for current_only, so a
    # leftover rehearsal state (hybrid + filter) would silently corrupt every baseline number.
    agent.set_retrieval(mode="minilm", current_only=False)
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                          timeout=120)
    if not args.baseline_only and not client.list_snapshots(config.COLLECTION):
        raise SystemExit("No snapshot on the cluster. Run `uv run python snapshot.py backup` "
                         "first — this script deletes points and needs a restore point.")
    fix2 = load("fix2_embedding")
    fix3 = load("fix3_hybrid")
    fix1 = load("fix1_dedup")

    print("=" * 64)
    print("BASELINE (weak dense, duplicates present)")
    print(f"  fix1 duplicate-rate@{K} (fix1 queries): {dup_rate(fix1, mode='minilm'):.2f}")
    print(f"  fix2 recall@{K} (lookalikes, MiniLM):     {recall_at_k(fix2, mode='minilm'):.2f}")
    n3, m3 = ndcg_mrr(fix3, mode="minilm")
    print(f"  fix3 NDCG@{K} / MRR (paraphrases, MiniLM): {n3:.2f} / {m3:.2f}")
    print(f"  cold open correct (MiniLM):               {cold_open_correct(mode='minilm')}")
    if args.baseline_only:
        return

    print("\n" + "=" * 64)
    print("FIX #1 — dedup the collection")
    before = client.count(config.COLLECTION).count
    dup_ids = find_duplicate_ids(client)
    client.delete(collection_name=config.COLLECTION, points_selector=dup_ids)
    after = client.count(config.COLLECTION).count
    print(f"  points {before} -> {after} (removed {len(dup_ids)})")
    print(f"  fix1 duplicate-rate@{K}: {dup_rate(fix1, mode='minilm'):.2f}  (was crowded, now unique)")

    print("\n" + "=" * 64)
    print("FIX #2 — migrate weak -> strong embedding (zero-downtime, measured first)")
    print(f"  fix2 recall@{K}: MiniLM {recall_at_k(fix2, mode='minilm'):.2f}  ->  bge {recall_at_k(fix2, mode='bge'):.2f}")
    print("  (on the full dex the 384d model can't separate the clone Pokemon; bge can)")

    print("\n" + "=" * 64)
    print("FIX #3 — hybrid + ColBERT rerank (baseline = bge, the fix #2 state)")
    nw, mw = ndcg_mrr(fix3, mode="bge")
    nh, mh = ndcg_mrr(fix3, mode="hybrid")
    print(f"  fix3 NDCG@{K}: bge {nw:.2f} -> hybrid {nh:.2f}")
    print(f"  fix3 MRR:     bge {mw:.2f} -> hybrid {mh:.2f}")
    print(f"  fix3 recall@{K}: bge {recall_at_k(fix3, mode='bge'):.2f} -> hybrid {recall_at_k(fix3, mode='hybrid'):.2f}")

    print("\n" + "=" * 64)
    print("COLD-OPEN CLOSE — is_current payload filter")
    try:
        client.create_payload_index(config.COLLECTION, "is_current", models.PayloadSchemaType.BOOL)
    except Exception as e:  # already indexed is fine; anything else must be visible
        print(f"  (create_payload_index: {e})")
    print(f"  cold open correct: hybrid {cold_open_correct(mode='hybrid')}  ->  "
          f"hybrid+filter {cold_open_correct(mode='hybrid', current_only=True)}")
    print("\nDone. Run `uv run python snapshot.py restore` to revert to the baseline.")


if __name__ == "__main__":
    main()
