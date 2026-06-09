# LlamaIndex RAG with traceAI Tracing and Evaluation

This example shows how to build a local-document RAG pipeline with
[LlamaIndex](https://github.com/run-llama/llama_index), trace every LLM and
retrieval call with [traceAI](https://github.com/future-agi/traceAI), and run
automated evaluations using Future AGI.

## What This Example Demonstrates

- Loading a local text document into a LlamaIndex `VectorStoreIndex`
- Querying the index with natural-language questions
- Automatic OpenTelemetry tracing of every LLM call and retrieval step via `traceAI-llamaindex`
- Attaching evaluation tags (`CONTEXT_ADHERENCE`, `CONTEXT_RELEVANCE`, `COMPLETENESS`) so Future AGI scores each response automatically
- Viewing traces and scores in the [Future AGI dashboard](https://app.futureagi.com)

## Prerequisites

- Python 3.10–3.13 (Python 3.14+ is not yet supported by `fi-instrumentation-otel`)
- An [OpenAI API key](https://platform.openai.com/api-keys) *(required)*
- A [Future AGI account](https://app.futureagi.com) *(optional — needed only for traces and eval scores)*

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key |
| `FI_API_KEY` | No | Future AGI API key — enables tracing |
| `FI_SECRET_KEY` | No | Future AGI secret key — enables tracing |


## Setup

```bash
# 1. Clone the cookbooks repo
git clone https://github.com/future-agi/cookbooks.git
cd cookbooks/llamaindex-rag-tracing-eval

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env and fill in your keys
```

## Run

```bash
python app.py
```

## Expected Output

```
Building index from data/sample_policy.txt ...

Question: How many days per week can employees work remotely?
Answer:   Eligible employees can work remotely up to 3 days per week.

Question: What is the monthly stipend for home office expenses?
Answer:   The company provides a $50/month stipend for home office costs.

Question: What are the core working hours?
Answer:   Core hours are 10:00 AM to 3:00 PM in the employee's local timezone.

Done. Traces and evaluations are visible at https://app.futureagi.com
```

After running, open [app.futureagi.com](https://app.futureagi.com), navigate
to the **llamaindex-rag-tracing-eval** project, and you will see:

- A span tree for each query: `query → retrieve → llm`
- Token counts, latency, and full prompt/response payloads
- Automated evaluation scores for Context Adherence, Context Relevance, and Completeness

## Project Structure

```
llamaindex-rag-tracing-eval/
├── app.py              # Main RAG pipeline with tracing
├── eval_tags.py        # Future AGI evaluation tag definitions
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── data/
    └── sample_policy.txt   # Sample document for the index
```

## How This Maps to Production RAG

| This Example | Production Equivalent |
|---|---|
| `SimpleDirectoryReader` | S3, GCS, or a document store loader |
| In-memory `VectorStoreIndex` | Pinecone, Qdrant, pgvector, etc. |
| `gpt-4o-mini` | Any model supported by LlamaIndex |
| `ProjectType.EXPERIMENT` | Use `ProjectType.OBSERVE` in production |
| Three hardcoded questions | User queries from an API or chat interface |

To switch to production mode, change `ProjectType.EXPERIMENT` to
`ProjectType.OBSERVE` in `app.py` and remove `eval_tags`. In observe mode,
traces are still captured but evaluation scoring is disabled.

## Troubleshooting

**`AuthenticationError` from Future AGI**
Verify `FI_API_KEY` and `FI_SECRET_KEY` in your `.env` file match what is
shown in the Future AGI dashboard under Settings → API Keys.

**`openai.AuthenticationError`**
Check that `OPENAI_API_KEY` is set correctly and has billing enabled.

**Traces not appearing in dashboard**
The exporter is async. Wait ~10 seconds after the script finishes. If still
missing, set `OTEL_BSP_SCHEDULE_DELAY=1000` in your `.env` to flush faster.

**`ModuleNotFoundError: traceai_llamaindex`**
Run `pip install traceAI-llamaindex` (note the capital A and I match the PyPI package name).

## Contact

For questions about this example open an issue in
[future-agi/cookbooks](https://github.com/future-agi/cookbooks/issues).
For traceAI SDK questions see
[future-agi/traceAI](https://github.com/future-agi/traceAI).