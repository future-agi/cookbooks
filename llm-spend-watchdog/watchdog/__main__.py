"""python -m watchdog {check,replay,digest}"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from . import tracing  # noqa: E402
from .agent import llm_available  # noqa: E402
from .run import load_ctx, check, replay, weekly_digest  # noqa: E402


def _at(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(prog="watchdog", description="LLM spend and budget watchdog")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "sample", help="folder with the usage sources and budget sheet")
    ap.add_argument("--month", default="2026-11", help="YYYY-MM of the month being watched")
    ap.add_argument("--no-llm", action="store_true", help="write messages from the template, not the agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="one hourly read")
    c.add_argument("--at", help="ISO time of the read (default: now)")
    c.add_argument("--approve", action="store_true", help="ask before posting to Slack (default: dry run)")
    c.add_argument("--reset", action="store_true", help="forget which alerts were already sent this month")
    sub.add_parser("replay", help="replay the month hour by hour (dry run)")
    d = sub.add_parser("digest", help="weekly summary")
    d.add_argument("--week-end", default="2026-11-23T09:00Z")
    d.add_argument("--approve", action="store_true")
    a = ap.parse_args()

    traced = tracing.init()
    print(f"Tracing to Future AGI: {'on' if traced else 'off (set FI_API_KEY and FI_SECRET_KEY)'}. "
          f"Messages written by: {'the agent' if llm_available() and not a.no_llm else 'the template'}.")
    ctx = load_ctx(a.data, a.month)
    if a.cmd == "check":
        check(ctx, _at(a.at), a.approve, not a.no_llm, a.reset)
    elif a.cmd == "replay":
        r = replay(ctx, not a.no_llm)
        print(f"\n{r['alerts']} alerts in {len(r['messages'])} messages; {r['false_alerts']} false.")
        for e in r["event_detail"]:
            mark = "caught" if e["caught"] else "MISSED"
            print(f"  {mark}: {e['event']}" + (f", {e['lead_hours']}h ahead" if e["caught"] else ""))
        print("Messages saved to outbox/replay/. Nothing was posted.")
    elif a.cmd == "digest":
        weekly_digest(ctx, _at(a.week_end), a.approve)
    tracing.flush()


if __name__ == "__main__":
    main()
