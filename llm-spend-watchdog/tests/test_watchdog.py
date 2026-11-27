import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.generate import Scenario, Spike, sample_scenario, write  # noqa: E402
from evals.cases import H  # noqa: E402
from watchdog.alerts import compose, redact  # noqa: E402
from watchdog.detect import State, hourly_read, top_spenders  # noqa: E402
from watchdog.run import load_ctx, replay, settings  # noqa: E402
from watchdog.sources import load_usage  # noqa: E402


def month(sc: Scenario):
    tmp = tempfile.TemporaryDirectory()
    truth = write(sc, Path(tmp.name))
    return tmp, truth, load_ctx(Path(tmp.name), "2026-11")


class SampleMonth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.truth, cls.ctx = month(sample_scenario())
        cls.result = replay(cls.ctx, use_llm=False, quiet=True, write_outbox=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_done_when(self):
        """The sheet's bar: runaway key and zero-balance day alerted before they happen, no false alerts."""
        events = {e["event"]: e for e in self.result["event_detail"]}
        self.assertTrue(events["key:oa-support-bot passes its budget"]["caught"])
        self.assertTrue(events["balance:mistral-prepaid hits zero"]["caught"])
        self.assertEqual(self.result["false_alerts"], 0)

    def test_no_secret_in_any_message(self):
        for m in self.result["messages"]:
            for s in self.truth["fake_secrets"]:
                self.assertNotIn(s, m["message"])
            self.assertNotIn("sk-", m["message"])

    def test_reads_never_look_ahead(self):
        """Changing the future must not change what an hourly read says now."""
        usage = load_usage(self.ctx.data, self.ctx.keys, self.ctx.month_start, self.ctx.hours_in_month)
        h = H(18, 20)
        future = copy.deepcopy(usage)
        for series in future.cost.values():
            for i in range(h, len(series)):
                series[i] *= 50
        a, b = State(), State()
        for hour in range(1, h + 1):
            fa = hourly_read(usage, self.ctx.keys, self.ctx.budgets, self.ctx.balances, a, settings(), h=hour)
            fb = hourly_read(future, self.ctx.keys, self.ctx.budgets, self.ctx.balances, b, settings(), h=hour)
            self.assertEqual([f.id for f in fa], [f.id for f in fb])


class Quiet(unittest.TestCase):
    def test_one_off_spike_is_not_projected(self):
        """A $30 one-hour spike that doesn't push the key over its budget: no projection alert."""
        tmp, _, ctx = month(Scenario("spike", 5, spikes=[Spike("oa-search", H(12, 10), 30.0)]))
        r = replay(ctx, use_llm=False, quiet=True, write_outbox=False)
        tmp.cleanup()
        self.assertFalse([e for e in r["event_detail"] if e["event"].startswith("key:oa-search")])
        self.assertEqual(r["false_alerts"], 0)
        self.assertFalse([m for m in r["messages"] for f in m["findings"] if f.kind == "projected_breach"])

    def test_one_line_per_scope_when_levels_jump(self):
        tmp, _, ctx = month(sample_scenario())
        usage = load_usage(ctx.data, ctx.keys, ctx.month_start, ctx.hours_in_month)
        found = hourly_read(usage, ctx.keys, ctx.budgets, ctx.balances, State(), settings(), h=H(23, 11))
        tmp.cleanup()
        ids = [f"{f.kind}|{f.scope_type}:{f.scope_id}" for f in found]
        self.assertEqual(len(ids), len(set(ids)))


class Redaction(unittest.TestCase):
    def test_patterns(self):
        text = "key sk-fake-abcdefgh12345678 and Bearer abcdefghijklmnop and api_key=supersecretvalue"
        out = redact(text)
        self.assertNotIn("sk-fake", out)
        self.assertNotIn("abcdefghijklmnop", out)
        self.assertNotIn("supersecretvalue", out)


class AgentTools(unittest.TestCase):
    def test_tools_return_json_without_secrets(self):
        os.environ.setdefault("LLM_MODEL", "anthropic/claude-haiku-5-5")
        from watchdog.agent import build_agent

        tmp, truth, ctx = month(sample_scenario())
        usage = load_usage(ctx.data, ctx.keys, ctx.month_start, ctx.hours_in_month)
        st = State()
        found = []
        for hour in range(1, H(18, 22) + 1):
            found = hourly_read(usage, ctx.keys, ctx.budgets, ctx.balances, st, settings(), h=hour) or found
        top = top_spenders(usage, ctx.keys, H(18, 22))
        agent = build_agent(found, top, usage, ctx.keys, H(18, 22))
        tools = agent.tools
        alerts = json.loads(tools["get_new_alerts"]())
        self.assertTrue(alerts)
        self.assertEqual(len(json.loads(tools["get_top_spenders"](5))), 5)
        daily = json.loads(tools["get_daily_spend"]("oa-support-bot", 3))
        self.assertEqual(len(daily["daily_usd"]), 3)
        blob = json.dumps(alerts) + compose(found, top, "now")
        for s in truth["fake_secrets"]:
            self.assertNotIn(s, blob)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
