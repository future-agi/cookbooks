"""Generate the scaling-decay curve (arc step 3): the same golden queries scored against
the ACTUAL broken pipeline (weak dense, fragmented + duplicated) as the Pokedex grows
generation by generation — Gen 1 only, through Gen 4, then the full dex. Camera never
opens this; the notebook loads its output.

Honest controlled experiment: the gold documents (all Gen 1) are present from the first
stage; growth adds real later-generation species — including genuinely similar ones
(Pikachu's ten lookalikes) — plus the same skewed re-crawl duplication as the real
ingest. Two things decay together: recall@5 falls and the duplicate rate in the top-k
rises, which the three fixes later reverse. The curve runs the broken pipeline only
(weak dense): the opening hook shows the decay, and each fix section — dedup, the bge
migration, hybrid+rerank — then claims its own slice of the recovery. Writes
data/scaling_curve.{json,png}.

    uv run python scaling_curve.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from helpers import config, embeddings
from helpers.corpus import build_corpus
from ingest import build_points, load_gold_doc_ids

# Growth stages: species introduced up to this generation are in the collection.
STAGE_GENS = [1, 4, 9]
SCALE_COLLECTION = "pokemon_scale"
DATA = Path(__file__).resolve().parent / "data"
K = config.TOP_K


def semantic_queries(pts):
    """Scored queries whose gold doc already exists at the first stage — a query whose
    answer document only shipped in a later generation can't hit in the Gen 1 era and
    would distort the decay comparison."""
    rows = [json.loads(l) for l in (DATA / "golden_dataset.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["exercises"] in ("fix2_embedding", "fix3_hybrid", "fix1_dedup")]
    stage1 = {p["doc_id"] for p in pts if max(p["intro_gen"], p["generation"]) <= STAGE_GENS[0]}
    kept = [r for r in rows if any(g in stage1 for g in r["gold_doc_ids"])]
    if len(kept) < len(rows):
        print(f"excluded {len(rows) - len(kept)} queries whose gold docs don't exist at gen<={STAGE_GENS[0]}")
    return kept


def main():
    load_dotenv()
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                          timeout=120)
    # Build the full broken point set once (fragmented flavor + skewed duplication),
    # then embed every unique text once and reuse across stages.
    gold_ids = load_gold_doc_ids()
    pts = build_points(build_corpus(), gold_ids)
    queries = semantic_queries(pts)
    print(f"built {len(pts)} broken points; embedding unique texts (weak)...")
    uniq = list({p["text"] for p in pts})
    vec = dict(zip(uniq, embeddings.dense(uniq, config.MODEL_DENSE_WEAK)))
    qvec = embeddings.dense([r["query"] for r in queries], config.MODEL_DENSE_WEAK, is_query=True)

    results = []
    for stage_gen in STAGE_GENS:
        # Real growth: each stage is the collection as it stood in that generation's era —
        # a document exists once its species is in the dex AND its facts have shipped
        # (a Gen-5 flavor entry for Pikachu did not exist in the Gen-1 era). Gold docs
        # are all Gen 1, so they are present from the first stage.
        pool = [p for p in pts if max(p["intro_gen"], p["generation"]) <= stage_gen]
        client.delete_collection(SCALE_COLLECTION)
        client.create_collection(SCALE_COLLECTION, vectors_config={
            config.DENSE_WEAK: models.VectorParams(size=config.DIM_DENSE_WEAK, distance=models.Distance.COSINE)})
        for s in range(0, len(pool), 256):
            batch = pool[s:s + 256]
            client.upsert(SCALE_COLLECTION, points=[
                models.PointStruct(id=s + i, vector={config.DENSE_WEAK: vec[p["text"]]},
                                   payload={"doc_id": p["doc_id"], "text": p["text"]})
                for i, p in enumerate(batch)])
        hit, dup = 0, 0.0
        for r, qv in zip(queries, qvec):
            top = client.query_points(SCALE_COLLECTION, query=qv, using=config.DENSE_WEAK,
                                      limit=K, with_payload=["doc_id", "text"],
                                      search_params=models.SearchParams(exact=True)).points
            docs = [(h.payload["doc_id"], h.payload["text"]) for h in top]
            hit += int(any(d[0] in set(r["gold_doc_ids"]) for d in docs))
            dup += 1 - len(set(docs)) / len(docs) if docs else 0
        results.append({"stage_gen": stage_gen, "size": len(pool),
                        "recall": hit / len(queries), "dup_rate": dup / len(queries)})
        print(f"  gen<={stage_gen}  size={len(pool):>6}  recall@{K}={results[-1]['recall']:.2f}"
              f"  dup_rate@{K}={results[-1]['dup_rate']:.2f}")

    client.delete_collection(SCALE_COLLECTION)
    (DATA / "scaling_curve.json").write_text(json.dumps(results, indent=2))

    xs = [r["size"] for r in results]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, [r["recall"] for r in results], "o-", color="#cc0000", label="recall@5 (baseline)")
    ax1.set_ylabel("recall@5", color="#cc0000"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["dup_rate"] for r in results], "s--", color="#e08a00", label="duplicate rate@5")
    ax2.set_ylabel("duplicate rate@5", color="#e08a00"); ax2.set_ylim(0, 1.05)
    ax1.set_xscale("log"); ax1.set_xticks(xs)
    ax1.set_xticklabels([f"Gen 1\n({r['size']:,} pts)" if r["stage_gen"] == 1
                         else f"Gen 1-{r['stage_gen']}\n({r['size']:,} pts)"
                         for r in results])
    ax1.set_xlabel("Pokedex coverage (collection size)")
    plt.title("Same golden queries, growing Pokedex — baseline decays")
    fig.tight_layout(); plt.savefig(DATA / "scaling_curve.png", dpi=130)
    print(f"wrote {DATA/'scaling_curve.json'} and scaling_curve.png")


if __name__ == "__main__":
    main()
