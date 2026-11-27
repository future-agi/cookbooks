"""traceAI: every hourly read is a trace in a Future AGI project.

On when FI_API_KEY and FI_SECRET_KEY are set. The hourly read is a CHAIN span;
the agent's steps, tool calls and LLM calls nest under it through the
smolagents instrumentor.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

_tracer = None


def init() -> bool:
    global _tracer
    if _tracer is not None:
        return True
    if not (os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY")):
        return False
    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_smolagents import SmolagentsInstrumentor

    provider = register(project_type=ProjectType.OBSERVE,
                        project_name=os.getenv("FI_PROJECT_NAME", "llm-spend-watchdog"),
                        set_global_tracer_provider=True, verbose=False)
    SmolagentsInstrumentor().instrument(tracer_provider=provider)
    _tracer = provider.get_tracer("llm-spend-watchdog")
    return True


@contextmanager
def span(name: str, kind: str = "CHAIN", **inputs):
    if _tracer is None:
        yield _Noop()
        return
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("fi.span.kind", kind)
        s.set_attribute("gen_ai.span.kind", kind)
        s.set_attribute("input.value", json.dumps(inputs, default=str))
        s.set_attribute("input.mime_type", "application/json")
        yield _Span(s)


class _Span:
    def __init__(self, s):
        self.s = s

    def output(self, value) -> None:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        self.s.set_attribute("output.value", text)

    def attr(self, key: str, value) -> None:
        self.s.set_attribute(key, value)


class _Noop:
    def output(self, value) -> None:
        pass

    def attr(self, key: str, value) -> None:
        pass


def flush() -> None:
    if _tracer is not None:
        from opentelemetry import trace

        p = trace.get_tracer_provider()
        if hasattr(p, "force_flush"):
            p.force_flush()
