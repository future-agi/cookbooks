"""Run the golden set through the traced agent — the Future AGI eval entry point.

Every question goes through agent.ask(), so each run exports full traces (agent,
retriever, and LLM spans). `--stage iterN` routes the run to that fix's own Future AGI
project (see STAGE_PROJECTS), so every before/after is one project switch on the
dashboard. Run it once per arc stage — baseline, after dedup, after the bge flip, after
hybrid, after the filter. The retrieval state is whatever the notebook last set; this
script never changes it.

Also writes data/golden_run-{mode}.jsonl (query, expected_answer, answer, retrieved
doc_ids, and retrieved_context — the exact text the model read, as one string, ready
for fi.evals.evaluate(context=...)). Full set takes a few minutes.

    uv run python run_golden.py                        # all 37 queries
    uv run python run_golden.py --group fix2_embedding # one slice
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from helpers import config

DATA = Path(__file__).resolve().parent / "data"

# One Future AGI project per arc stage: the dashboard compares fixes by switching project.
STAGE_PROJECTS = {
    "iter1": "pokedex-webinar",
    "iter2": "pokedex-webinar-fix1-dedup",
    "iter3": "pokedex-webinar-fix2-bge",
    "iter4": "pokedex-webinar-fix3-hybrid",
    "iter5": "pokedex-webinar-fix4-filter",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", help="only one slice, e.g. fix2_embedding")
    ap.add_argument("--stage", help="iteration label (iter1, iter2, ...); names the output "
                                    "file and selects the run's Future AGI project")
    args = ap.parse_args()
    if args.group and args.stage:
        ap.error("--group overwrites the stage's full run file with a slice; "
                 "use --group alone for ad-hoc checks, --stage alone for scored runs")
    if args.stage:
        os.environ["FI_STAGE_TAG"] = args.stage
        if args.stage in STAGE_PROJECTS:
            # Assign, don't setdefault: a leftover FI_PROJECT_NAME in the shell must not
            # route a scored stage into the wrong Future AGI project.
            os.environ["FI_PROJECT_NAME"] = STAGE_PROJECTS[args.stage]
    # Import AFTER the stage label is set: the agent registers tracing at import, and the
    # stage rides on register(metadata=...) — project-level, applied to all spans.
    import agent

    rows = [json.loads(l) for l in (DATA / "golden_dataset.jsonl").read_text().splitlines()
            if l.strip()]
    if args.group:
        rows = [r for r in rows if r["exercises"] == args.group]

    state = agent.retrieval_state()
    label = args.stage or (state["mode"] + ("-current" if state["current_only"] else ""))
    out = DATA / f"golden_run-{label}.jsonl"
    with out.open("w") as f:
        for i, r in enumerate(rows, 1):
            answer, chunks = agent.ask(r["query"], expected_answer=r["expected_answer"])
            f.write(json.dumps({
                "query": r["query"], "exercises": r["exercises"],
                "expected_answer": r["expected_answer"], "answer": answer,
                "retrieved_doc_ids": [c["doc_id"] for c in chunks],
                # exactly what the model read (all hops), as the one string FI evals expect
                "retrieved_context": agent.get_model_context(),
            }) + "\n")
            print(f"[{i}/{len(rows)}] {r['query'][:60]}")
    print(f"wrote {out.name}; traces exported to Future AGI project "
          f"{os.getenv('FI_PROJECT_NAME', 'pokedex-rag')}")


if __name__ == "__main__":
    main()
