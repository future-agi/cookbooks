"""Run the watchdog over the 24 synthetic months and score it.

Future AGI evals (ai-evaluation SDK, run locally, no key needed):
- numeric_similarity: projected month-end spend vs what was really spent,
  for keys with nothing injected, read on days 8, 15 and 22
- numeric_similarity: hours until a budget is passed, as the first
  projected_breach alert said vs as it happened
- numeric_similarity: projected hours until a balance hits zero (at the first
  runout alert) vs the real hours
- contains_none: no alert message contains any of the month's API secrets or
  anything shaped like one

With --cloud (needs FI_API_KEY and FI_SECRET_KEY), every message is also
scored by Future AGI's Turing judge for groundedness and is_concise.

Scored by this suite directly from the ground truth (see watchdog/score.py):
alert precision (no false or duplicate alerts) and event recall (budget
breaches and empty balances warned before they happen).

    python -m evals.run_evals [--llm] [--cloud]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from statistics import mean

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from fi.evals import evaluate  # noqa: E402

from data.generate import write  # noqa: E402
from evals.cases import CASES, H  # noqa: E402
from watchdog import tracing  # noqa: E402
from watchdog.detect import members, parse_when, project, top_spenders  # noqa: E402
from watchdog.run import load_ctx, replay  # noqa: E402

CHECKPOINTS = (H(8), H(15), H(22))
LEAK_WORDS = ["sk-", "Bearer "]


def touched(sc) -> set[str]:
    return {r.key for r in sc.runaways} | {s.key for s in sc.spikes} | {g.key for g in sc.growth}


def numbers_grounded(message: str, findings, top) -> float:
    """Share of the numbers in the message that appear in the facts it was given."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?( UTC)?", " ", message)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)]
    nums = [n.rstrip(".") for n in nums if n.rstrip(".")]
    if not nums:
        return 1.0
    allowed: set[float] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 24.0}  # list numbering, "top 5", "last 6 hours", "7 days", "24h"
    for f in findings:
        for v in list(f.facts.values()) + [f.level]:
            if isinstance(v, (int, float)) and v is not None:
                allowed |= {round(float(v), 2), round(float(v), 1), round(float(v)), round(float(v) * 100, 1), round(float(v) * 100)}
    for r in top:
        for v in (r["spend_24h"], r["spend_mtd"]):
            allowed |= {round(v, 2), round(v, 1), round(v)}
    ok = sum(1 for n in nums if any(abs(float(n) - a) < 0.011 for a in allowed))
    return ok / len(nums)


def run_case(sc, use_llm: bool, cloud: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        truth = write(sc, Path(tmp))
        ctx = load_ctx(Path(tmp), "2026-11")
        with tracing.span("eval.case", case=sc.name):
            r = replay(ctx, use_llm, quiet=True, write_outbox=False)
    usage = r["usage"]

    # Projection error, Future AGI numeric_similarity.
    proj_scores = []
    skip = touched(sc)
    for b in ctx.budgets:
        if b.scope_type != "key" or b.scope_id in skip:
            continue
        actual = r["month_end"][f"key:{b.scope_id}"]
        for cp in CHECKPOINTS:
            p = project(usage, members("key", b.scope_id, ctx.keys), cp)["projected_eom"]
            proj_scores.append(evaluate("numeric_similarity", output=f"{p:.4f}", expected_output=f"{actual:.4f}").score)

    runout_score = None
    zero = r["zero_hour"].get("balance:mistral-prepaid")
    first = next((f for m in r["messages"] for f in m["findings"] if f.kind == "runout"), None)
    if zero is not None and first is not None:
        actual_h = zero - parse_when(first.at, usage)
        runout_score = evaluate("numeric_similarity", output=str(first.facts["hours_until_zero"]), expected_output=f"{actual_h:.1f}").score

    # Breach timing, Future AGI numeric_similarity: hours until the budget is
    # passed, as the first projected_breach alert said vs as it happened.
    eta_scores = []
    for e in r["event_detail"]:
        if not (e["caught"] and e["event"].endswith("passes its budget")):
            continue
        scope = e["event"].split(" ")[0]
        f = next(f for m in r["messages"] for f in m["findings"]
                 if f.kind == "projected_breach" and f"{f.scope_type}:{f.scope_id}" == scope)
        actual_h = e["happens_hour"] - parse_when(f.at, usage)
        eta_scores.append(evaluate("numeric_similarity", output=str(f.facts["hours_until_breach"]), expected_output=f"{actual_h:.1f}").score)

    leak_scores, grounded, cloud_scores = [], [], {"groundedness": [], "is_concise": []}
    secrets = truth["fake_secrets"]
    for m in r["messages"]:
        leak_scores.append(evaluate("contains_none", output=m["message"], keywords=secrets + LEAK_WORDS).score)
        top = top_spenders(usage, ctx.keys, int(parse_when(m["at"], usage)), 5)
        grounded.append(numbers_grounded(m["message"], m["findings"], top))
        if cloud:
            facts = json.dumps({"alerts": [{"kind": f.kind, "scope": f.scope_id, **f.facts} for f in m["findings"]], "top": top})
            for name in cloud_scores:
                try:
                    res = evaluate(name, output=m["message"], context=facts, input="Write the Slack alert for these findings.", model="turing_flash")
                    if res.score is not None:
                        cloud_scores[name].append(float(res.score))
                except Exception as e:  # keep going; report what ran
                    print(f"  {name} failed on {sc.name}: {e}")

    return {
        "case": sc.name,
        "alerts": r["alerts"],
        "false_alerts": r["false_alerts"],
        "precision": r["precision"],
        "events": r["events"],
        "caught_before": r["caught_before"],
        "recall": r["recall"],
        "min_lead_hours": min((e["lead_hours"] for e in r["event_detail"] if e["caught"]), default=None),
        "projection_similarity": round(mean(proj_scores), 4) if proj_scores else None,
        "runout_similarity": None if runout_score is None else round(runout_score, 4),
        "breach_eta_similarity": round(mean(eta_scores), 4) if eta_scores else None,
        "no_key_leak": round(mean(leak_scores), 4) if leak_scores else 1.0,
        "numbers_grounded": round(mean(grounded), 4) if grounded else 1.0,
        "written_by": sorted({m["written_by"] for m in r["messages"]}),
        "cloud": {k: round(mean(v), 4) for k, v in cloud_scores.items() if v},
        "wrong": r["wrong"],
        "missed": [e["event"] for e in r["event_detail"] if not e["caught"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="let the agent write the messages (needs an LLM key)")
    ap.add_argument("--cloud", action="store_true", help="also run Future AGI Turing evals on the messages")
    a = ap.parse_args()
    tracing.init()
    if a.cloud and not (os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY")):
        sys.exit("--cloud needs FI_API_KEY and FI_SECRET_KEY in .env")

    rows = []
    for sc in CASES:
        row = run_case(sc, a.llm, a.cloud)
        rows.append(row)
        print(f"{row['case']:<30} alerts {row['alerts']:>2}  false {row['false_alerts']}  caught {row['caught_before']}/{row['events']}"
              f"  proj {row['projection_similarity']}  eta {row['breach_eta_similarity']}  runout {row['runout_similarity']}  leak-free {row['no_key_leak']}")
    tracing.flush()

    def avg(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return round(mean(vals), 4) if vals else None

    total_alerts = sum(r["alerts"] for r in rows)
    total_false = sum(r["false_alerts"] for r in rows)
    total_events = sum(r["events"] for r in rows)
    total_caught = sum(r["caught_before"] for r in rows)
    summary = {
        "cases": len(rows),
        "alerts": total_alerts,
        "false_alerts": total_false,
        "alert_precision": round((total_alerts - total_false) / total_alerts, 4) if total_alerts else 1.0,
        "events": total_events,
        "caught_before": total_caught,
        "event_recall": round(total_caught / total_events, 4) if total_events else 1.0,
        "projection_similarity": avg("projection_similarity"),
        "runout_similarity": avg("runout_similarity"),
        "breach_eta_similarity": avg("breach_eta_similarity"),
        "no_key_leak": avg("no_key_leak"),
        "numbers_grounded": avg("numbers_grounded"),
        "written_by": sorted({w for r in rows for w in r["written_by"]}),
    }
    if a.cloud:
        for k in ("groundedness", "is_concise"):
            vals = [r["cloud"][k] for r in rows if k in r["cloud"]]
            summary[k] = round(mean(vals), 4) if vals else None
    out = ROOT / "evals" / "results.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
