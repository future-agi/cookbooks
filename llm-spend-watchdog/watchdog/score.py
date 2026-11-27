"""Was each alert right, and did the ones that matter come in time?

An alert is right when what it claims is true over the whole month:
- threshold: spend really had crossed that share of the budget
- projected_breach: the budget really is passed before the month ends
- runout: the balance really hits zero within twice the warning window
- balance_empty: the balance really is at zero
A second alert for the same scope, kind and level is spam and counts as wrong.

The events that must be caught: every budget that is passed during the month
needs a projected_breach alert before it is passed, and every balance that
hits zero needs a runout alert before it does.
"""

from __future__ import annotations

from .detect import Finding, actual_crossings, members, parse_when, spend


def score(timeline: list[Finding], usage, keys, budgets, balances, cfg) -> dict:
    truth = actual_crossings(usage, keys, budgets, balances)
    budget_of = {f"{b.scope_type}:{b.scope_id}": b for b in budgets}
    seen, results = set(), []
    for f in timeline:
        h = parse_when(f.at, usage)
        sk = f"{f.scope_type}:{f.scope_id}"
        ok, why = True, ""
        if f.id in seen:
            ok, why = False, "duplicate"
        elif f.kind == "threshold":
            b = budget_of[sk]
            ok = spend(usage, members(b.scope_type, b.scope_id, keys), 0, int(h)) >= f.level * b.monthly_budget_usd - 1e-6
            why = "" if ok else "threshold not crossed"
        elif f.kind == "projected_breach":
            ok = truth["breach_hour"].get(sk) is not None
            why = "" if ok else "budget not passed by month end"
        elif f.kind == "runout":
            z = truth["zero_hour"].get(sk)
            ok = z is not None and z - h <= 2 * cfg.runout_warn_hours
            why = "" if ok else "balance did not run out soon"
        elif f.kind == "balance_empty":
            z = truth["zero_hour"].get(sk)
            ok = z is not None and z <= h
        seen.add(f.id)
        results.append({"at": f.at, "id": f.id, "correct": ok, "why": why})

    events = []
    for sk, hit in truth["breach_hour"].items():
        if hit is None:
            continue
        first = [parse_when(f.at, usage) for f in timeline if f.kind == "projected_breach" and f"{f.scope_type}:{f.scope_id}" == sk]
        warned = min(first) if first else None
        events.append({"event": f"{sk} passes its budget", "happens_hour": hit, "warned_hour": warned,
                       "caught": warned is not None and warned < hit, "lead_hours": None if warned is None else round(hit - warned, 1)})
    for sk, hit in truth["zero_hour"].items():
        if hit is None:
            continue
        first = [parse_when(f.at, usage) for f in timeline if f.kind == "runout" and f"{f.scope_type}:{f.scope_id}" == sk]
        warned = min(first) if first else None
        events.append({"event": f"{sk} hits zero", "happens_hour": hit, "warned_hour": warned,
                       "caught": warned is not None and warned < hit, "lead_hours": None if warned is None else round(hit - warned, 1)})

    n = len(results)
    wrong = [r for r in results if not r["correct"]]
    caught = [e for e in events if e["caught"]]
    return {
        "alerts": n,
        "false_alerts": len(wrong),
        "precision": 1.0 if n == 0 else round((n - len(wrong)) / n, 4),
        "events": len(events),
        "caught_before": len(caught),
        "recall": 1.0 if not events else round(len(caught) / len(events), 4),
        "wrong": wrong,
        "event_detail": events,
        "month_end": truth["month_end"],
        "zero_hour": truth["zero_hour"],
    }
