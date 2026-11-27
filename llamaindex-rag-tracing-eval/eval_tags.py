from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ModelChoices,
)

list_of_eval_tags = [
    EvalTag(
        eval_name=EvalName.CONTEXT_ADHERENCE,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "context": "input.value",
            "output": "output.value",
        },
        custom_eval_name="Context Adherence",
        model=ModelChoices.TURING_LARGE,
    ),
    EvalTag(
        eval_name=EvalName.CONTEXT_RELEVANCE,
        value=EvalSpanKind.RETRIEVER,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "input": "input.value",
            "context": "output.value",
        },
        custom_eval_name="Context Relevance",
        model=ModelChoices.TURING_LARGE,
    ),
    EvalTag(
        eval_name=EvalName.COMPLETENESS,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        config={},
        mapping={
            "input": "input.value",
            "output": "output.value",
        },
        custom_eval_name="Completeness",
        model=ModelChoices.TURING_LARGE,
    ),
]
