"""24 synthetic months. Each one is a case the watchdog has to get right.

Some must raise an alert ahead of time (a runaway key, slow growth that passes
a budget, a balance running out). Others must stay quiet about projections
(a one-off spike, a retry loop fixed within a few hours).
"""

from data.generate import Growth, Runaway, Scenario, Spike, sample_scenario


def H(day: int, hour: int = 0) -> int:
    """Hour index for day (1-based) and hour of day, UTC."""
    return (day - 1) * 24 + hour


CASES: list[Scenario] = [
    sample_scenario(),
    # Runaway keys: must be warned before the budget is passed.
    Scenario("runaway-early-oa-evals", 101, runaways=[Runaway("oa-evals", H(5, 10), 1.5)]),
    Scenario("runaway-oa-search-day8", 102, runaways=[Runaway("oa-search", H(8, 16), 3.0)]),
    Scenario("runaway-an-sales-day12", 103, runaways=[Runaway("an-sales", H(12, 3), 2.0)]),
    Scenario("runaway-gm-voice-day22", 104, runaways=[Runaway("gm-voice", H(22, 9), 4.0)]),
    Scenario("runaway-late-an-coding-day26", 105, runaways=[Runaway("an-coding", H(26, 13), 6.0)]),
    Scenario("runaway-stops-after-30h", 106, runaways=[Runaway("oa-support-bot", H(14, 8), 6.0, duration_hours=30)]),
    Scenario("two-runaways", 107, runaways=[Runaway("gm-extract", H(9, 12), 1.2), Runaway("an-docs-qa", H(16, 20), 3.5)]),
    Scenario("runaway-and-balance-day18", 108, runaways=[Runaway("oa-search", H(11, 6), 2.5)], balance_zero_hour=H(18, 9)),
    # Short retry loops someone fixed: no projected alert.
    Scenario("burst-3h-fixed", 109, runaways=[Runaway("oa-support-bot", H(15, 10), 8.0, duration_hours=3)]),
    Scenario("burst-2h-fixed", 110, runaways=[Runaway("an-sales", H(10, 14), 15.0, duration_hours=2)]),
    Scenario("burst-4h-fixed", 111, runaways=[Runaway("gm-voice", H(19, 1), 5.0, duration_hours=4)]),
    # One-off spikes (a backfill, a big eval run): no projected alert.
    Scenario("spike-40-an-coding", 112, spikes=[Spike("an-coding", H(10, 11), 40.0)]),
    Scenario("spike-60-oa-search", 113, spikes=[Spike("oa-search", H(20, 15), 60.0)]),
    Scenario("backfill-2x25-gm-batch", 114, spikes=[Spike("gm-batch", H(6, 2), 25.0), Spike("gm-batch", H(6, 3), 25.0)]),
    # Prepaid balance running out on different days.
    Scenario("balance-zero-day12", 115, balance_zero_hour=H(12, 18)),
    Scenario("balance-zero-day20", 116, balance_zero_hour=H(20, 4)),
    Scenario("balance-zero-day29", 117, balance_zero_hour=H(29, 22)),
    # Slow growth.
    Scenario("growth-an-docs-qa-6pct", 118, growth=[Growth("an-docs-qa", H(10), 0.06)]),
    Scenario("growth-gm-extract-4pct", 119, growth=[Growth("gm-extract", H(5), 0.04)]),
    Scenario("growth-oa-search-1pct", 120, growth=[Growth("oa-search", H(3), 0.01)]),
    # Quiet months.
    Scenario("quiet-month-a", 121),
    Scenario("quiet-month-b", 122, spikes=[Spike("mi-triage", H(7, 9), 20.0)]),
    Scenario("quiet-month-c", 123, spikes=[Spike("oa-evals", H(13, 22), 15.0), Spike("an-sales", H(24, 5), 18.0)]),
]

assert len(CASES) >= 20 and len({c.name for c in CASES}) == len(CASES)
