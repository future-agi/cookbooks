from fi_instrumentation.fi_types import EvalName, EvalTag, EvalTagType, EvalSpanKind

list_of_eval_tags = [
    EvalTag(
        eval_name=EvalName.CONTEXT_ADHERENCE,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "context": "raw.input",
            "output": "raw.output",
        },
        custom_eval_name="Context Adherence",
    ),
    EvalTag(
        eval_name=EvalName.CONTEXT_RELEVANCE,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "input": "raw.input",
            "context": "raw.output",
        },
        custom_eval_name="Context Relevance",
    ),
    EvalTag(
        eval_name=EvalName.COMPLETENESS,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "input": "raw.input",
            "output": "raw.output",
        },
        custom_eval_name="Completeness",
    ),
]