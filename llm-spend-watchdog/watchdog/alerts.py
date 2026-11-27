"""Turning findings into a Slack message, without an LLM, and keeping secrets out.

The agent (agent.py) writes the message when an LLM key is set; this template
is the fallback and the reference the evals compare it against.
"""

from __future__ import annotations

import re

from .detect import Finding

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*\S{8,}"),
]


def redact(text: str) -> str:
    """Last line of defence: nothing that looks like a credential leaves the process."""
    for p in SECRET_PATTERNS:
        text = p.sub("[redacted]", text)
    return text


URGENCY = {"balance_empty": 0, "runout": 1, "projected_breach": 2, "threshold": 3}


def _money(v: float) -> str:
    return f"${v:,.2f}"


def finding_line(f: Finding) -> str:
    x = f.facts
    who = f"*{f.scope_id}*" + (f" ({f.scope_type})" if f.scope_type != "key" else "") + f", owner {f.owner_team}"
    if f.kind == "threshold":
        return f"{who}: passed {int(f.level * 100)}% of its {_money(x['budget_usd'])} monthly budget ({_money(x['spend_mtd_usd'])}, {x['pct_of_budget']}%)."
    if f.kind == "projected_breach":
        return (f"{who}: on course to pass its {_money(x['budget_usd'])} budget around {x['projected_breach_at']} "
                f"(in {x['hours_until_breach']}h). {x['pct_of_budget']}% used; spending {_money(x['last_6h_per_day_usd'])}/day "
                f"over the last 6 hours; the last 7 days averaged {_money(x['last_7d_avg_per_day_usd'])}/day.")
    if f.kind == "runout":
        return (f"*{f.scope_id}* balance: {_money(x['remaining_usd'])} left, projected to run out around "
                f"{x['projected_zero_at']} (in {x['hours_until_zero']}h) at {_money(x['burn_per_day_usd'])}/day.")
    if f.kind == "balance_empty":
        return f"*{f.scope_id}* balance is used up: requests on {', '.join(x['keys'])} will start failing."
    return f"{who}: {f.kind}"


def compose(findings: list[Finding], top: list[dict], at: str) -> str:
    fs = sorted(findings, key=lambda f: URGENCY.get(f.kind, 9))
    lines = [f"*LLM spend watchdog* · {at}", ""]
    lines += [f"• {finding_line(f)}" for f in fs]
    lines += ["", "*Top 5 keys, last 24h*"]
    lines += [f"{i}. {r['key']} · {r['project']} · {r['provider']} · {_money(r['spend_24h'])} (month to date {_money(r['spend_mtd'])})"
              for i, r in enumerate(top, 1)]
    urgent = [f for f in fs if f.kind != "threshold"]
    if urgent:
        lines += ["", "Next step for a person: check the keys above for a loop or a traffic change, or top up the balance. "
                      "The watchdog only reads usage; it never tops up or turns off a key."]
    return redact("\n".join(lines))


def digest(facts: dict) -> str:
    w = facts
    lines = [f"*Weekly LLM spend* · {w['week_start']} to {w['week_end']}", "",
             f"Total {_money(w['total'])} ({w['change_pct']:+.0f}% on the week before, {_money(w['previous_total'])})."]
    lines += ["", "*By provider*"] + [f"• {p}: {_money(v)}" for p, v in w["by_provider"].items()]
    lines += ["", "*Top 5 keys this week*"] + [f"{i}. {r['key']} · {r['project']} · {_money(r['week'])}" for i, r in enumerate(w["top"], 1)]
    lines += ["", "*Budgets, month to date*"] + [
        f"• {b['scope']}: {_money(b['spend'])} of {_money(b['budget'])} ({b['pct']}%), projected {_money(b['projected'])} by month end"
        for b in w["budgets"]]
    for bal in w["balances"]:
        lines.append(f"• {bal['id']} balance: {_money(bal['remaining'])} left" + (f", runs out around {bal['zero_at']}" if bal.get("zero_at") else ""))
    return redact("\n".join(lines))
