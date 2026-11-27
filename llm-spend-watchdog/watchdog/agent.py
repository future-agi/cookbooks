"""The agent: reads the hour's findings through tools and writes the Slack message.

Detection stays in detect.py (deterministic, evaluated on its own). The LLM's
job is the part people read: lead with what's urgent, say who owns it, give
the top spenders, and suggest a next step for a person. It has read-only
tools; nothing it does is sent anywhere until a person approves it.
"""

from __future__ import annotations

import json
import os

from smolagents import LiteLLMModel, ToolCallingAgent, tool

from .alerts import redact
from .detect import Finding, members, spend
from .sources import Key, Usage

INSTRUCTIONS = """You are the LLM spend watchdog for an engineering team.
Write ONE Slack message (Slack mrkdwn) about this hour's new spend alerts.

Rules:
- Call get_new_alerts first. Use only numbers the tools return; never estimate your own.
- Lead with the most urgent: a balance that is empty or about to run out, then budgets projected to be passed, then thresholds already crossed.
- One bullet per alert: the key, project, provider or balance by its alias, its owner team, what happened or is projected and when (UTC), and the numbers behind it.
- Then "Top 5 keys, last 24h" from get_top_spenders(5).
- End with one suggested next step for a person. Never say you topped up, paused or cut off anything: you only read usage.
- Never print an API key, token or secret. Refer to keys only by their alias, like oa-support-bot.
- Keep it under 1,500 characters. Return the message with final_answer.
"""


def llm_available() -> bool:
    return bool(os.getenv("LLM_MODEL")) and any(
        os.getenv(v) for v in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"))


def build_agent(findings: list[Finding], top: list[dict], usage: Usage, keys: list[Key], h: int) -> ToolCallingAgent:
    by_scope = {"key": "key", "provider": "provider", "all": "org"}

    @tool
    def get_new_alerts() -> str:
        """Returns this hour's new spend alerts as JSON. Each has kind (threshold, projected_breach,
        runout, balance_empty), scope_type, scope_id, level, owner_team and the numbers behind it."""
        return json.dumps([{"kind": f.kind, "scope_type": f.scope_type, "scope_id": f.scope_id,
                            "level": f.level, "owner_team": f.owner_team, **f.facts} for f in findings])

    @tool
    def get_top_spenders(n: int) -> str:
        """Returns the keys that spent the most in the last 24 hours, with month-to-date spend.

        Args:
            n: How many keys to return, 1 to 10.
        """
        return json.dumps(top[: max(1, min(int(n), 10))])

    @tool
    def get_daily_spend(scope_id: str, days: int) -> str:
        """Returns daily spend in USD for a key alias, a provider name, or 'all', oldest day first.

        Args:
            scope_id: A key alias such as oa-support-bot, a provider such as openai, or all.
            days: How many days back, 1 to 14.
        """
        kind = "key" if any(k.key_id == scope_id for k in keys) else by_scope.get(scope_id, "provider")
        ids = members(kind, scope_id, keys)
        if not ids:
            return json.dumps({"error": f"unknown scope {scope_id}"})
        days = max(1, min(int(days), 14))
        out = []
        for d in range(days, 0, -1):
            start, end = max(0, h - d * 24), max(0, h - (d - 1) * 24)
            out.append(round(spend(usage, ids, start, end), 2))
        return json.dumps({"scope": scope_id, "daily_usd": out})

    model = LiteLLMModel(model_id=os.environ["LLM_MODEL"], api_key=os.getenv("LLM_API_KEY") or None)
    return ToolCallingAgent(tools=[get_new_alerts, get_top_spenders, get_daily_spend], model=model,
                            instructions=INSTRUCTIONS, max_steps=6, verbosity_level=0)


def write_message(findings: list[Finding], top: list[dict], usage: Usage, keys: list[Key], h: int, at: str) -> str:
    agent = build_agent(findings, top, usage, keys, h)
    out = agent.run(f"Hourly read at {at}: {len(findings)} new alert(s). Write the Slack message.")
    return redact(str(out))
