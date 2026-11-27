"""The hourly read: which budgets crossed a threshold, which will, and which
balance runs out soon. Pure functions over Usage, no LLM.

Burn rate per key is the larger of two robust rates, so a one-hour spike
doesn't raise an alarm and a sustained runaway does within a few hours:

- recent: the last 24 hours with the 4 most expensive hours left out
- weekly: the last 7 days with the top 3% of hours left out (keeps nightly
  batch jobs and weekends in the average)

When a prepaid balance runs out depends on which days are ahead (weekdays
cost more), so its runout time walks forward hour by hour using the same hour
last week, scaled by how the last 24 hours compare with the same 24 hours a
week earlier. With under a week of history it falls back to the burn rate.

A projected alert (budget breach or balance runout) also has to hold for
`persist_reads` hourly reads in a row before it fires. Each alert fires once
per scope and level per month.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .sources import Balance, Budget, Key, Usage


@dataclass
class Settings:
    thresholds: tuple[float, ...] = (0.5, 0.8, 1.0)
    runout_warn_hours: int = 72
    persist_reads: int = 3
    top_n: int = 5
    min_history_hours: int = 24


@dataclass
class Finding:
    kind: str  # threshold | projected_breach | runout | balance_empty
    scope_type: str
    scope_id: str
    level: float | None
    at: str
    owner_team: str
    facts: dict

    @property
    def id(self) -> str:
        return f"{self.kind}|{self.scope_type}:{self.scope_id}|{self.level}"


@dataclass
class State:
    fired: dict[str, str] = field(default_factory=dict)  # finding id -> when
    pending: dict[str, int] = field(default_factory=dict)  # finding id -> reads in a row

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> State:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


def members(scope_type: str, scope_id: str, keys: list[Key]) -> list[str]:
    attr = {"key": "key_id", "project": "project", "team": "team", "provider": "provider"}.get(scope_type)
    if scope_type == "org":
        return [k.key_id for k in keys]
    return [k.key_id for k in keys if getattr(k, attr) == scope_id]


def _trimmed_mean(values: list[float], drop: int) -> float:
    if not values:
        return 0.0
    drop = min(drop, len(values) - 1)
    kept = sorted(values)[: len(values) - drop]
    return sum(kept) / len(kept)


def key_rate(series: list[float], h: int) -> float:
    """$ per hour, from hours before h only."""
    recent = _trimmed_mean(series[max(0, h - 24) : h], 4)
    week = series[max(0, h - 168) : h]
    weekly = _trimmed_mean(week, max(1, round(len(week) * 0.03)))
    return max(recent, weekly)


def hours_until(usage: Usage, ids: list[str], h: int, amount: float, horizon: int) -> float | None:
    """Hours from h until `amount` more is spent, or None if not within horizon."""
    rate = sum(key_rate(usage.cost[k], h) for k in ids)
    if h < 168 + 24:
        return amount / rate if rate > 0 and amount / rate <= horizon else None
    now = spend(usage, ids, h - 24, h)
    week_ago = spend(usage, ids, h - 192, h - 168)
    trend = min(3.0, max(0.5, now / week_ago)) if week_ago > 0 else 1.0
    run = 0.0
    for i in range(min(horizon, 168)):  # never reads an hour at or after h
        step = sum(usage.cost[k][h + i - 168] for k in ids) * trend
        if run + step >= amount:
            return i + (amount - run) / step
        run += step
    return None


def spend(usage: Usage, ids: list[str], start: int, end: int) -> float:
    return sum(sum(usage.cost[k][start:end]) for k in ids)


def project(usage: Usage, ids: list[str], h: int) -> dict:
    mtd = spend(usage, ids, 0, h)
    rate = sum(key_rate(usage.cost[k], h) for k in ids)
    left = usage.hours_in_month - h
    # The robust rate decides whether to alert; the last 6 hours' pace says when,
    # since a loop that's running now spends at today's pace, not the week's.
    # The 2 dearest of those hours are left out, so a backfill doesn't set it.
    last6 = [sum(usage.cost[k][i] for k in ids) for i in range(max(0, h - 6), h)]
    pace = max(rate, _trimmed_mean(last6, 2))
    return {"mtd": mtd, "rate_per_hour": rate, "pace_per_hour": pace, "projected_eom": mtd + rate * left}


def _when(usage: Usage, h: float) -> str:
    return (usage.month_start + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M UTC")


def top_spenders(usage: Usage, keys: list[Key], h: int, n: int = 5) -> list[dict]:
    rows = []
    for k in keys:
        last24 = spend(usage, [k.key_id], max(0, h - 24), h)
        rows.append({"key": k.key_id, "project": k.project, "provider": k.provider, "team": k.team,
                     "spend_24h": round(last24, 2), "spend_mtd": round(spend(usage, [k.key_id], 0, h), 2)})
    rows.sort(key=lambda r: -r["spend_24h"])
    return rows[:n]


def hourly_read(usage: Usage, keys: list[Key], budgets: list[Budget], balances: list[Balance],
                state: State, cfg: Settings = Settings(), h: int | None = None) -> list[Finding]:
    """New findings at hour h (default: the latest known hour). Updates state."""
    h = usage.hours if h is None else h
    at = _when(usage, h)
    found: list[Finding] = []
    seen_pending: set[str] = set()

    def fire(f: Finding, persist: bool) -> None:
        if f.id in state.fired:
            return
        if persist:
            seen_pending.add(f.id)
            state.pending[f.id] = state.pending.get(f.id, 0) + 1
            if state.pending[f.id] < cfg.persist_reads:
                return
        state.fired[f.id] = at
        found.append(f)

    for b in budgets:
        ids = members(b.scope_type, b.scope_id, keys)
        if not ids or b.monthly_budget_usd <= 0:
            continue
        p = project(usage, ids, h)
        pct = p["mtd"] / b.monthly_budget_usd
        base = {"budget_usd": round(b.monthly_budget_usd, 2), "spend_mtd_usd": round(p["mtd"], 2),
                "pct_of_budget": round(pct * 100, 1), "burn_per_day_usd": round(p["rate_per_hour"] * 24, 2),
                "projected_month_end_usd": round(p["projected_eom"], 2), "keys": ids if b.scope_type != "key" else None}
        crossed = [lv for lv in cfg.thresholds if pct >= lv]
        if crossed:
            # Several levels crossed since the last read: one line, for the highest.
            for lv in crossed[:-1]:
                state.fired.setdefault(Finding("threshold", b.scope_type, b.scope_id, lv, at, "", {}).id, at)
            fire(Finding("threshold", b.scope_type, b.scope_id, crossed[-1], at, b.owner_team, dict(base)), persist=False)
        if h >= cfg.min_history_hours and pct < 1.0 and p["projected_eom"] > b.monthly_budget_usd and p["rate_per_hour"] > 0:
            eta_h = h + (b.monthly_budget_usd - p["mtd"]) / p["pace_per_hour"]
            week = min(h, 168)
            facts = dict(base, last_6h_per_day_usd=round(p["pace_per_hour"] * 24, 2),
                         last_7d_avg_per_day_usd=round(spend(usage, ids, h - week, h) / week * 24, 2),
                         projected_breach_at=_when(usage, eta_h), hours_until_breach=round(eta_h - h, 1))
            fire(Finding("projected_breach", b.scope_type, b.scope_id, 1.0, at, b.owner_team, facts), persist=True)

    for bal in balances:
        ids = members("provider", bal.provider, keys)
        start = max(0, int((bal.as_of - usage.month_start).total_seconds() // 3600))
        used = spend(usage, ids, start, h)
        left = bal.starting_balance_usd - used
        rate = sum(key_rate(usage.cost[k], h) for k in ids)
        base = {"starting_balance_usd": round(bal.starting_balance_usd, 2), "remaining_usd": round(max(left, 0), 2),
                "burn_per_day_usd": round(rate * 24, 2), "keys": ids}
        if left <= 0:
            fire(Finding("balance_empty", "balance", bal.balance_id, 0.0, at, "finance", base), persist=False)
        elif h >= cfg.min_history_hours and rate > 0 and (hz := hours_until(usage, ids, h, left, cfg.runout_warn_hours)) is not None:
            facts = dict(base, projected_zero_at=_when(usage, h + hz), hours_until_zero=round(hz, 1))
            fire(Finding("runout", "balance", bal.balance_id, None, at, "finance", facts), persist=True)

    for fid in list(state.pending):
        if fid not in seen_pending and fid not in state.fired:
            del state.pending[fid]  # condition broke: start counting again
    return found


def actual_crossings(usage: Usage, keys: list[Key], budgets: list[Budget], balances: list[Balance]) -> dict:
    """Ground truth from the full month: when each budget hit 100% and each balance hit 0."""
    out = {"breach_hour": {}, "zero_hour": {}, "month_end": {}}
    for b in budgets:
        ids = members(b.scope_type, b.scope_id, keys)
        run = 0.0
        hit = None
        for h in range(usage.hours_in_month):
            run += sum(usage.cost[k][h] for k in ids)
            if hit is None and run >= b.monthly_budget_usd:
                hit = h + 1
        out["breach_hour"][f"{b.scope_type}:{b.scope_id}"] = hit
        out["month_end"][f"{b.scope_type}:{b.scope_id}"] = run
    for bal in balances:
        ids = members("provider", bal.provider, keys)
        run, hit = 0.0, None
        for h in range(usage.hours_in_month):
            run += sum(usage.cost[k][h] for k in ids)
            if hit is None and run >= bal.starting_balance_usd:
                hit = h + 1
        out["zero_hour"][f"balance:{bal.balance_id}"] = hit
    return out


def parse_when(s: str, usage: Usage) -> float:
    t = datetime.strptime(s, "%Y-%m-%d %H:%M UTC").replace(tzinfo=usage.month_start.tzinfo)
    return (t - usage.month_start).total_seconds() / 3600
