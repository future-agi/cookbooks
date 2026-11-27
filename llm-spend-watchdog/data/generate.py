"""Synthetic LLM usage for one month.

Four providers, twelve keys. The sample month has one runaway key (a retry
loop that starts on day 18) and one prepaid balance that hits zero on day 25.

Usage is written the way the watchdog reads it in production:

- gateway_logs.jsonl       one row per request through the gateway (Google, Mistral keys)
- openai_costs.json        hourly cost buckets, grouped by API key (OpenAI keys)
- anthropic_cost_report.json  hourly cost report, amounts in cents (Anthropic keys)
- keys.csv, budgets.csv, balances.csv   the budget sheet's three tabs
- truth.json               what was injected, for the evals

Every value is made up. The "secrets" in request headers are fake and exist so
the evals can check that no alert ever prints one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

MONTH_START = datetime(2026, 11, 1, tzinfo=timezone.utc)
DAYS = 30
HOURS = DAYS * 24

# key_id, provider, source, project, team, $ per day on a normal weekday
KEYS = [
    ("oa-support-bot", "openai", "openai_api", "support-bot", "support", 9.0),
    ("oa-search", "openai", "openai_api", "search", "platform", 6.0),
    ("oa-evals", "openai", "openai_api", "evals-pipeline", "research", 4.0),
    ("an-sales", "anthropic", "anthropic_api", "sales-assistant", "gtm", 7.0),
    ("an-docs-qa", "anthropic", "anthropic_api", "docs-qa", "platform", 5.0),
    ("an-coding", "anthropic", "anthropic_api", "internal-coding", "eng", 8.0),
    ("gm-extract", "google", "gateway", "data-extract", "data", 3.0),
    ("gm-voice", "google", "gateway", "voice-agent", "support", 4.0),
    ("gm-batch", "google", "gateway", "nightly-batch", "data", 2.5),
    ("mi-triage", "mistral", "gateway", "ticket-triage", "support", 6.0),
    ("mi-summaries", "mistral", "gateway", "call-summaries", "gtm", 9.0),
    ("mi-translate", "mistral", "gateway", "translation", "platform", 8.0),
]

MODELS = {
    "openai": ("gpt-5-mini", 0.6),
    "anthropic": ("claude-haiku-5-5", 1.2),
    "google": ("gemini-3-flash", 0.4),
    "mistral": ("mistral-medium-3", 0.8),
}  # model, $ per 1M blended tokens (made up)


@dataclass
class Runaway:
    key: str
    start_hour: int  # hour index from the month start
    extra_per_hour: float  # $ added every hour while it runs
    duration_hours: int | None = None  # None: runs to the end of the month


@dataclass
class Spike:
    key: str
    hour: int
    amount: float


@dataclass
class Growth:
    key: str
    from_hour: int
    daily_rate: float  # 0.05 = +5% a day, compounding


@dataclass
class Scenario:
    name: str = "sample-month"
    seed: int = 7
    runaways: list[Runaway] = field(default_factory=list)
    spikes: list[Spike] = field(default_factory=list)
    growth: list[Growth] = field(default_factory=list)
    balance_zero_hour: float | None = None  # when the Mistral prepaid balance hits zero
    note: str = ""


def sample_scenario() -> Scenario:
    return Scenario(
        name="sample-month",
        seed=7,
        runaways=[Runaway("oa-support-bot", start_hour=17 * 24 + 14, extra_per_hour=5.5)],
        spikes=[Spike("an-coding", hour=9 * 24 + 11, amount=38.0)],
        balance_zero_hour=24 * 24 + 15,  # day 25, 15:00 UTC
        note="Retry loop on oa-support-bot from day 18 14:00; one-off batch spike on an-coding day 10; "
        "Mistral prepaid balance runs out on day 25.",
    )


def _hour_weight(hod: int) -> float:
    return 0.4 + 1.2 * math.exp(-(((hod - 14) / 4.5) ** 2))


_W = [_hour_weight(h) for h in range(24)]
_W_MEAN = sum(_W) / 24


def hourly_costs(sc: Scenario) -> dict[str, list[float]]:
    """$ spent per key per hour. Deterministic for a given scenario."""
    rng = random.Random(sc.seed)
    out: dict[str, list[float]] = {}
    for key_id, _provider, _src, _proj, _team, per_day in KEYS:
        drift = rng.uniform(-0.004, 0.004)
        series = []
        for h in range(HOURS):
            ts = MONTH_START + timedelta(hours=h)
            hod, day = ts.hour, h // 24
            weekend = 0.55 if ts.weekday() >= 5 else 1.0
            base = per_day / 24 * (1 + drift) ** day * weekend
            if key_id == "gm-batch":
                shaped = base * 0.6 * _W[hod] / _W_MEAN + (base * 24 * 0.4 / 3 if hod in (2, 3, 4) else 0.0)
            else:
                shaped = base * _W[hod] / _W_MEAN
            series.append(shaped * rng.lognormvariate(0, 0.2))
        out[key_id] = series

    for g in sc.growth:
        s = out[g.key]
        for h in range(g.from_hour, HOURS):
            s[h] *= (1 + g.daily_rate) ** ((h - g.from_hour) / 24)
    for r in sc.runaways:
        end = HOURS if r.duration_hours is None else min(HOURS, r.start_hour + r.duration_hours)
        for h in range(r.start_hour, end):
            out[r.key][h] += r.extra_per_hour * rng.uniform(0.85, 1.15)
    for sp in sc.spikes:
        out[sp.key][sp.hour] += sp.amount
    return out


def budgets_for(costs: dict[str, list[float]], sc: Scenario) -> list[dict]:
    """Monthly budgets sized so a normal month ends at 55-80% of budget."""
    rng = random.Random(sc.seed + 1)
    normal = hourly_costs(Scenario(seed=sc.seed))  # same noise, nothing injected
    rows = []
    for key_id, provider, _src, _proj, team, _ in KEYS:
        if provider == "mistral":
            continue  # prepaid balance instead of a budget
        eom = sum(normal[key_id])
        rows.append({"scope_type": "key", "scope_id": key_id, "monthly_budget_usd": round(eom / rng.uniform(0.55, 0.8), -1), "owner_team": team})
    for provider in ("openai", "anthropic", "google"):
        eom = sum(sum(normal[k[0]]) for k in KEYS if k[1] == provider)
        rows.append({"scope_type": "provider", "scope_id": provider, "monthly_budget_usd": round(eom / 0.7, -1), "owner_team": "finance"})
    eom_all = sum(sum(v) for v in normal.values())
    rows.append({"scope_type": "org", "scope_id": "all", "monthly_budget_usd": round(eom_all / 0.72, -2), "owner_team": "finance"})
    return rows


def balance_for(costs: dict[str, list[float]], sc: Scenario) -> list[dict]:
    mistral = [k[0] for k in KEYS if k[1] == "mistral"]
    if sc.balance_zero_hour is None:
        start = round(sum(sum(costs[k]) for k in mistral) * 1.6, -1)  # plenty left at month end
    else:
        whole = int(sc.balance_zero_hour)
        start = sum(costs[k][h] for k in mistral for h in range(whole))
        start += (sc.balance_zero_hour - whole) * sum(costs[k][whole] for k in mistral)
        start = round(start, 2)
    return [{"balance_id": "mistral-prepaid", "provider": "mistral", "starting_balance_usd": start, "as_of": MONTH_START.isoformat()}]


def _fake_secret(rng: random.Random, prefix: str) -> str:
    return prefix + "".join(rng.choice("abcdefghijkmnopqrstuvwxyz0123456789") for _ in range(32))


def write(sc: Scenario, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    costs = hourly_costs(sc)
    rng = random.Random(sc.seed + 2)
    meta = {k[0]: k for k in KEYS}
    fake_secrets = {k[0]: _fake_secret(rng, "sk-fake-") for k in KEYS}

    with open(out / "keys.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key_id", "provider", "source", "source_ref", "project", "team"])
        for key_id, provider, src, proj, team, _ in KEYS:
            ref = {"openai_api": "key_" + key_id.replace("-", "_"), "anthropic_api": "apikey_" + key_id.replace("-", "_")}.get(src, "gk_" + key_id.replace("-", "_"))
            w.writerow([key_id, provider, src, ref, proj, team])
    refs = {r["key_id"]: r["source_ref"] for r in csv.DictReader(open(out / "keys.csv"))}

    budgets = budgets_for(costs, sc)
    with open(out / "budgets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope_type", "scope_id", "monthly_budget_usd", "owner_team"])
        w.writeheader()
        w.writerows(budgets)
    balances = balance_for(costs, sc)
    with open(out / "balances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["balance_id", "provider", "starting_balance_usd", "as_of"])
        w.writeheader()
        w.writerows(balances)

    # Gateway: one row per request.
    with open(out / "gateway_logs.jsonl", "w") as f:
        for key_id, provider, src, proj, team, _ in KEYS:
            if src != "gateway":
                continue
            model, per_m = MODELS[provider]
            for h, cost in enumerate(costs[key_id]):
                n = max(1, round(cost / 0.4))
                parts = [rng.random() + 0.2 for _ in range(n)]
                tot = sum(parts)
                for p in parts:
                    c = cost * p / tot
                    tokens = int(c / per_m * 1_000_000)
                    ts = MONTH_START + timedelta(hours=h, seconds=rng.randrange(3600))
                    row = {
                        "request_id": f"req_{rng.getrandbits(64):016x}",
                        "model": model,
                        "provider": provider,
                        "started_at": ts.isoformat().replace("+00:00", "Z"),
                        "input_tokens": int(tokens * 0.8),
                        "output_tokens": tokens - int(tokens * 0.8),
                        "total_tokens": tokens,
                        "cost": f"{c:.6f}",
                        "status_code": 200,
                        "api_key_id": refs[key_id],
                        "user_id": f"svc-{proj}",
                        "metadata": {"project": proj, "team": team},
                        "request_headers": {"x-provider-key": fake_secrets[key_id]},
                    }
                    f.write(json.dumps(row) + "\n")

    # OpenAI-style hourly cost buckets.
    buckets = []
    for h in range(HOURS):
        start = int((MONTH_START + timedelta(hours=h)).timestamp())
        results = [
            {"object": "organization.costs.result", "amount": {"value": round(costs[k][h], 6), "currency": "usd"}, "line_item": None, "project_id": "proj_" + meta[k][3].replace("-", "_"), "api_key_id": refs[k]}
            for k in costs if meta[k][2] == "openai_api"
        ]
        buckets.append({"object": "bucket", "start_time": start, "end_time": start + 3600, "results": results})
    json.dump({"object": "page", "data": buckets, "has_more": False}, open(out / "openai_costs.json", "w"))

    # Anthropic-style cost report, amounts in cents as strings.
    report = []
    for h in range(HOURS):
        s = MONTH_START + timedelta(hours=h)
        results = [
            {"currency": "USD", "amount": f"{costs[k][h] * 100:.4f}", "workspace_id": "wrkspc_" + meta[k][4], "api_key_id": refs[k], "description": meta[k][3]}
            for k in costs if meta[k][2] == "anthropic_api"
        ]
        report.append({"starting_at": s.isoformat().replace("+00:00", "Z"), "ending_at": (s + timedelta(hours=1)).isoformat().replace("+00:00", "Z"), "results": results})
    json.dump({"data": report, "has_more": False, "next_page": None}, open(out / "anthropic_cost_report.json", "w"))

    truth = {
        "scenario": asdict(sc),
        "month_start": MONTH_START.isoformat(),
        "hours": HOURS,
        "fake_secrets": sorted(fake_secrets.values()),
    }
    json.dump(truth, open(out / "truth.json", "w"), indent=2)
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "sample")
    args = ap.parse_args()
    sc = sample_scenario()
    write(sc, args.out)
    print(f"Wrote the synthetic month to {args.out}: {sc.note}")


if __name__ == "__main__":
    main()
