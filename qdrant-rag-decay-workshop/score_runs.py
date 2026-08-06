"""Score the golden runs on Future AGI evals and print the scoreboard. Camera never opens this.

`run_golden.py --stage iterN` records what the agent answered; this reads those files back
and prints the table the notebook shows, so a re-run can never leave the notebook quoting
scores the platform no longer agrees with. Per-row scores land in data/scores-<stage>.jsonl,
which is where to look when a mean moves and you want to know which query moved it. Re-running
resumes from whatever the last run finished, so an eval outage costs no completed rows.

    uv run python score_runs.py                 # every data/golden_run-iter*.jsonl
    uv run python score_runs.py iter1 iter2     # only these stages
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before the judge model is read below

DATA = Path(__file__).resolve().parent / "data"
JUDGE = "turing_flash"
ATTEMPTS = 6  # backoff sums to ~63s, enough to ride out the endpoint's short brownouts

# The diagnostic triad plus the two generation-side checks. All five score the answer against
# the context the model read, so none of them can see a stale-but-relevant document.
EVALS = ["context_relevance", "chunk_utilization", "groundedness",
         "chunk_attribution", "detect_hallucination"]

# Answer vs expected_answer — the only eval holding ground truth, and so the only one that
# catches the cold open. It is a custom judge because no built-in fits: factual_accuracy takes
# input/output/context, so it grades against the retrieved document and scores a confident
# answer from the stale type chart 1.0; ground_truth_match wants the answer to match the
# reference's wording, so it fails a correct answer both for being shorter and for adding
# detail. The beat needs "same core claim, extra grounded detail allowed".
GROUND_TRUTH = "answer_correctness"
CRITERIA = (
    "Decide whether the answer is factually correct, using the reference answer as ground truth.\n"
    "Score 1 if the answer commits to the same core claim as the reference. Extra correct detail "
    "is fine, and a shorter answer is fine as long as it does not contradict the reference.\n"
    "Score 0 if the answer contradicts the reference, or refuses to answer, or lists conflicting "
    "possibilities without committing to the reference's claim as the current fact. The question "
    "asks for one fact, so an answer that says it depends is not a correct answer.\n\n"
    "QUESTION: {input}\nANSWER: {output}\nREFERENCE ANSWER: {expected_answer}"
)
# A stronger model than the generator, so the judge is not grading its own work.
JUDGE_LLM = os.getenv("JUDGE_LLM", "anthropic/claude-sonnet-5")


def _retry(call) -> list:
    """Run one eval call, retrying the endpoint's intermittent 500s.

    A missing score is the only reliable failure signal: on a 500 the SDK still reports
    status="completed" with error=None and puts the server error in `reason`. Calls stay
    sequential — at four in flight the endpoint dropped a quarter of them."""
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            results = list(call())
            if all(r.score is not None for r in results):
                return results
        except Exception as e:  # noqa: BLE001 — any failure is worth another attempt
            last_error = e
        time.sleep(2 ** attempt)
    if last_error is not None:
        print(f"last error: {type(last_error).__name__}: {last_error}")
    raise SystemExit(f"eval endpoint failed {ATTEMPTS} times in a row; stopping — "
                     "rows scored so far are saved, re-run to resume")


def score_row(row: dict) -> dict:
    from fi.evals import evaluate

    scores = {r.eval_name: r.score for r in
              _retry(lambda: evaluate(EVALS, input=row["query"], output=row["answer"],
                                      context=row["retrieved_context"], model=JUDGE))}
    judged = _retry(lambda: [evaluate(GROUND_TRUTH, prompt=CRITERIA, engine="llm",
                                      model=JUDGE_LLM, input=row["query"], output=row["answer"],
                                      expected_answer=row["expected_answer"])])[0]
    return {"query": row["query"], "exercises": row["exercises"], **scores,
            GROUND_TRUTH: judged.score, "reason": (judged.reason or "")[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", help="stage labels; default = every golden_run-iter*")
    args = ap.parse_args()
    stages = args.stages or sorted(p.stem.removeprefix("golden_run-")
                                   for p in DATA.glob("golden_run-iter*.jsonl"))
    if not stages:
        raise SystemExit("No data/golden_run-iter*.jsonl. Run run_golden.py --stage iterN first.")

    columns = EVALS + [GROUND_TRUTH]
    table = []
    for stage in stages:
        rows = [json.loads(l) for l in
                (DATA / f"golden_run-{stage}.jsonl").read_text().splitlines() if l.strip()]
        # Append each row as it is scored and skip what a previous run finished. The eval
        # endpoint goes down for minutes at a time, and re-scoring 37 rows to recover the
        # one that failed wastes calls that are themselves the scarce resource.
        out = DATA / f"scores-{stage}.jsonl"
        scored = ([json.loads(l) for l in out.read_text().splitlines() if l.strip()]
                  if out.exists() else [])
        # Resume assumes the score file is a positional prefix of the run file. If the run
        # was regenerated (different queries or order), scoring must restart from scratch.
        for i, s in enumerate(scored):
            if s["query"] != rows[i]["query"]:
                raise SystemExit(f"{out.name} row {i} does not match {stage}'s run file — "
                                 f"the run was regenerated; delete {out.name} and re-run")
        with out.open("a") as f:
            for row in rows[len(scored):]:
                s = score_row(row)
                f.write(json.dumps(s) + "\n")
                f.flush()
                scored.append(s)
        # Mean, then the count of rows below the pass line: a mean over 37 rows shifts 0.03
        # when a single query flips, so the count is what says whether anything really moved.
        cells = [f"{sum(s[c] for s in scored) / len(scored):.2f} ({sum(s[c] < 0.5 for s in scored)})"
                 for c in columns]
        table.append(f"| {stage} | {len(scored)} | " + " | ".join(cells) + " |")
        print(f"scored {len(scored)} rows -> {out.name}")

    print("\nmean (rows scoring below 0.5)\n")
    print("| run | n | " + " | ".join(c.replace("_", " ") for c in columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    print("\n".join(table))


if __name__ == "__main__":
    main()
