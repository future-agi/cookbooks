"""Read usage from the three sources and the budget sheet.

Each loader turns its source into the same thing: $ per key per hour. The
files under data/ stand in for the live calls; swap a loader for an HTTP
client and nothing downstream changes. Only read-only credentials are ever
needed, and no loader keeps a secret it sees (request headers are dropped).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Key:
    key_id: str  # the alias used everywhere, never the secret
    provider: str
    source: str
    source_ref: str
    project: str
    team: str


@dataclass(frozen=True)
class Budget:
    scope_type: str  # key | project | team | provider | org
    scope_id: str
    monthly_budget_usd: float
    owner_team: str


@dataclass(frozen=True)
class Balance:
    balance_id: str
    provider: str
    starting_balance_usd: float
    as_of: datetime


@dataclass
class Usage:
    """$ per key per hour from month_start. hours = how many hours are known."""

    month_start: datetime
    hours_in_month: int
    hours: int
    cost: dict[str, list[float]]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _csv_rows(path: Path, sheet_url: str | None = None) -> list[dict]:
    if sheet_url:  # a Google Sheet tab published as CSV
        with urllib.request.urlopen(sheet_url, timeout=20) as r:
            return list(csv.DictReader(io.StringIO(r.read().decode())))
    return list(csv.DictReader(open(path)))


def load_keys(data: Path, sheet_url: str | None = None) -> list[Key]:
    return [Key(**r) for r in _csv_rows(data / "keys.csv", sheet_url)]


def load_budgets(data: Path, sheet_url: str | None = None) -> list[Budget]:
    return [
        Budget(r["scope_type"], r["scope_id"], float(r["monthly_budget_usd"]), r.get("owner_team", ""))
        for r in _csv_rows(data / "budgets.csv", sheet_url)
    ]


def load_balances(data: Path, sheet_url: str | None = None) -> list[Balance]:
    return [
        Balance(r["balance_id"], r["provider"], float(r["starting_balance_usd"]), _ts(r["as_of"]))
        for r in _csv_rows(data / "balances.csv", sheet_url)
    ]


def _add(cost: dict, key: str, h: int, amount: float, n: int) -> None:
    if 0 <= h < n:
        cost.setdefault(key, [0.0] * n)[h] += amount


def load_usage(data: Path, keys: list[Key], month_start: datetime, hours_in_month: int, until: datetime | None = None) -> Usage:
    """Everything spent before `until` (default: the whole month)."""
    by_ref = {k.source_ref: k.key_id for k in keys}
    n = hours_in_month
    cost: dict[str, list[float]] = {k.key_id: [0.0] * n for k in keys}

    def hour_of(ts: datetime) -> int:
        return int((ts - month_start).total_seconds() // 3600)

    # Gateway request logs.
    gw = data / "gateway_logs.jsonl"
    if gw.exists():
        for line in open(gw):
            row = json.loads(line)
            key = by_ref.get(row["api_key_id"])
            if key is None:
                continue
            _add(cost, key, hour_of(_ts(row["started_at"])), float(row["cost"]), n)

    # OpenAI-style cost buckets.
    oa = data / "openai_costs.json"
    if oa.exists():
        for b in json.load(open(oa))["data"]:
            h = hour_of(datetime.fromtimestamp(b["start_time"], tz=timezone.utc))
            for r in b["results"]:
                key = by_ref.get(r.get("api_key_id"))
                if key:
                    _add(cost, key, h, float(r["amount"]["value"]), n)

    # Anthropic-style cost report (cents).
    an = data / "anthropic_cost_report.json"
    if an.exists():
        for b in json.load(open(an))["data"]:
            h = hour_of(_ts(b["starting_at"]))
            for r in b["results"]:
                key = by_ref.get(r.get("api_key_id"))
                if key:
                    _add(cost, key, h, float(r["amount"]) / 100, n)

    known = n if until is None else max(0, min(n, hour_of(until)))
    for series in cost.values():
        for h in range(known, n):
            series[h] = 0.0
    return Usage(month_start, n, known, cost)
