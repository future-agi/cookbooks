"""The three things the watchdog does: an hourly check, a weekly digest, and a
replay of a whole month (the demo and the evals)."""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import tracing
from .agent import llm_available, write_message
from .alerts import compose, digest as digest_text
from .detect import Finding, Settings, State, hourly_read, hours_until, key_rate, members, project, spend, top_spenders
from .score import score
from .slack import deliver, save
from .sources import load_balances, load_budgets, load_keys, load_usage

ROOT = Path(__file__).resolve().parent.parent


def settings() -> Settings:
    return Settings(
        thresholds=tuple(float(x) for x in os.getenv("THRESHOLDS", "0.5,0.8,1.0").split(",")),
        runout_warn_hours=int(os.getenv("RUNOUT_WARN_HOURS", "72")),
        persist_reads=int(os.getenv("PERSIST_READS", "3")),
    )


@dataclass
class Ctx:
    data: Path
    month_start: datetime
    hours_in_month: int
    keys: list
    budgets: list
    balances: list


def load_ctx(data: Path, month: str) -> Ctx:
    y, m = (int(x) for x in month.split("-"))
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    hours = calendar.monthrange(y, m)[1] * 24
    sheet = os.getenv
    return Ctx(data, start, hours, load_keys(data, sheet("KEYS_SHEET_CSV_URL")),
               load_budgets(data, sheet("BUDGETS_SHEET_CSV_URL")), load_balances(data, sheet("BALANCES_SHEET_CSV_URL")))


def message_for(findings: list[Finding], usage, ctx: Ctx, h: int, at: str, use_llm: bool) -> tuple[str, str]:
    top = top_spenders(usage, ctx.keys, h, 5)
    if use_llm and llm_available():
        try:
            return write_message(findings, top, usage, ctx.keys, h, at), "agent"
        except Exception as e:  # the alert still goes out, from the template
            print(f"Agent failed ({type(e).__name__}: {e}); using the template.")
    return compose(findings, top, at), "template"


def check(ctx: Ctx, at: datetime, approve: bool, use_llm: bool, reset: bool = False) -> list[Finding]:
    usage = load_usage(ctx.data, ctx.keys, ctx.month_start, ctx.hours_in_month, until=at)
    state_path = ROOT / ".state" / f"state-{ctx.month_start:%Y-%m}.json"
    state_path.parent.mkdir(exist_ok=True)
    if reset:
        state_path.unlink(missing_ok=True)
    first_run = not state_path.exists()
    state, cfg, h = State.load(state_path), settings(), usage.hours
    with tracing.span("watchdog.hourly_read", at=at.isoformat()) as sp:
        # A first run reads the last few hours too, so a projection can already
        # count as sustained instead of waiting for three more hourly runs.
        hours = range(max(1, h - cfg.persist_reads + 1), h + 1) if first_run else [h]
        findings = [f for hh in hours for f in hourly_read(usage, ctx.keys, ctx.budgets, ctx.balances, state, cfg, h=hh)]
        state.save(state_path)
        at_s = f"{at:%Y-%m-%d %H:%M} UTC"
        if not findings:
            print(f"{at_s}: no new alerts.")
            sp.output({"alerts": 0})
            return []
        msg, by = message_for(findings, usage, ctx, usage.hours, at_s, use_llm)
        outcome = deliver(msg, f"alert-{at:%Y%m%d-%H%M}", approve=approve)
        sp.attr("watchdog.alerts", len(findings))
        sp.attr("watchdog.written_by", by)
        sp.attr("watchdog.delivery", outcome)
        sp.output(msg)
    return findings


def replay(ctx: Ctx, use_llm: bool, quiet: bool = False, write_outbox: bool = True) -> dict:
    usage = load_usage(ctx.data, ctx.keys, ctx.month_start, ctx.hours_in_month)
    cfg, state = settings(), State()
    timeline: list[Finding] = []
    messages: list[dict] = []
    with tracing.span("watchdog.replay", month=f"{ctx.month_start:%Y-%m}", data=str(ctx.data)) as root:
        for h in range(1, ctx.hours_in_month + 1):
            found = hourly_read(usage, ctx.keys, ctx.budgets, ctx.balances, state, cfg, h=h)
            if not found:
                continue
            at = found[0].at
            with tracing.span("watchdog.hourly_read", at=at, alerts=len(found)) as sp:
                msg, by = message_for(found, usage, ctx, h, at, use_llm)
                sp.attr("watchdog.written_by", by)
                sp.output(msg)
            timeline += found
            messages.append({"at": at, "message": msg, "written_by": by, "findings": found})
            if write_outbox:
                save(msg, "replay/" + at.replace(" UTC", "").replace(" ", "-").replace(":", ""))
            if not quiet:
                for f in found:
                    print(f"{at}  {f.kind:<17} {f.scope_type}:{f.scope_id}" + (f"  {int(f.level * 100)}%" if f.kind == "threshold" else ""))
        result = score(timeline, usage, ctx.keys, ctx.budgets, ctx.balances, cfg)
        root.output({k: result[k] for k in ("alerts", "false_alerts", "precision", "events", "caught_before", "recall")})
    result["messages"] = messages
    result["usage"] = usage
    return result


def weekly_digest(ctx: Ctx, week_end: datetime, approve: bool) -> str:
    usage = load_usage(ctx.data, ctx.keys, ctx.month_start, ctx.hours_in_month, until=week_end)
    h = usage.hours
    week, prev = (max(0, h - 168), h), (max(0, h - 336), max(0, h - 168))
    total = spend(usage, [k.key_id for k in ctx.keys], *week)
    previous = spend(usage, [k.key_id for k in ctx.keys], *prev)
    providers = sorted({k.provider for k in ctx.keys})
    top = sorted(({"key": k.key_id, "project": k.project, "week": round(spend(usage, [k.key_id], *week), 2)} for k in ctx.keys),
                 key=lambda r: -r["week"])[:5]
    budgets = []
    for b in ctx.budgets:
        p = project(usage, members(b.scope_type, b.scope_id, ctx.keys), h)
        budgets.append({"scope": f"{b.scope_id}" + ("" if b.scope_type == "key" else f" ({b.scope_type})"), "spend": p["mtd"],
                        "budget": b.monthly_budget_usd, "pct": round(100 * p["mtd"] / b.monthly_budget_usd, 1), "projected": p["projected_eom"]})
    budgets.sort(key=lambda r: -r["projected"] / r["budget"])
    balances = []
    for bal in ctx.balances:
        ids = members("provider", bal.provider, ctx.keys)
        left = bal.starting_balance_usd - spend(usage, ids, 0, h)
        rate = sum(key_rate(usage.cost[k], h) for k in ids)
        zero = None
        if left > 0 and rate > 0:
            hz = hours_until(usage, ids, h, left, 168) or left / rate
            zero = f"{usage.month_start + timedelta(hours=h + hz):%Y-%m-%d %H:%M} UTC"
        balances.append({"id": bal.balance_id, "remaining": max(left, 0), "zero_at": zero})
    facts = {"week_start": f"{usage.month_start + timedelta(hours=week[0]):%b %d}", "week_end": f"{week_end:%b %d}",
             "total": total, "previous_total": previous, "change_pct": 0 if previous == 0 else 100 * (total - previous) / previous,
             "by_provider": {p: spend(usage, members("provider", p, ctx.keys), *week) for p in providers},
             "top": top, "budgets": budgets, "balances": balances}
    with tracing.span("watchdog.weekly_digest", week_end=week_end.isoformat()) as sp:
        msg = digest_text(facts)
        sp.output(msg)
        deliver(msg, f"digest-{week_end:%Y%m%d}", approve=approve)
    return msg
