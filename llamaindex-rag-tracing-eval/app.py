import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# --- Optional tracing: only activates when FI_API_KEY and FI_SECRET_KEY are set ---
fi_api_key = os.environ.get("FI_API_KEY")
fi_secret_key = os.environ.get("FI_SECRET_KEY")

if fi_api_key and fi_secret_key:
    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_llamaindex import LlamaIndexInstrumentor
    from eval_tags import list_of_eval_tags

    trace_provider = register(
        project_type=ProjectType.EXPERIMENT,
        project_name="llamaindex-rag-tracing-eval",
        project_version_name="v1",
        eval_tags=list_of_eval_tags,
    )
    LlamaIndexInstrumentor().instrument(tracer_provider=trace_provider)
    print("traceAI instrumentation enabled — traces will appear at https://app.futureagi.com")
else:
    print("FI_API_KEY / FI_SECRET_KEY not set — running without traceAI instrumentation.")

# --- LlamaIndex settings ---
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def build_index(data_dir: str = "data") -> VectorStoreIndex:
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


def query_rag(index: VectorStoreIndex, question: str) -> str:
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question)
    return str(response)


if __name__ == "__main__":
    print("Building index from data/sample_policy.txt ...")
    index = build_index()

    questions = [
        "How many days per week can employees work remotely?",
        "What is the monthly stipend for home office expenses?",
        "What are the core working hours?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = query_rag(index, q)
        print(f"Answer:   {answer}")

    print("\nDone.")
    if fi_api_key and fi_secret_key:
        print("Traces and evaluations are visible at https://app.futureagi.com")