# ============================================================================
# IMPORTS
# ============================================================================
import os
import yaml
import argparse
import gradio as gr
from pathlib import Path
from typing import List, Any, Dict
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="Document Chat Assistant with configurable models and prompts")
parser.add_argument("--llm", default="gpt4o-mini", 
                    help="LLM model to use from config.yaml. Options: gpt4o-mini, gpt4, gpt35")
parser.add_argument("--embedding", default="large", 
                    help="Embedding model to use from config.yaml. Options: large, small, ada")
parser.add_argument("--qa-prompt", default="default", 
                    help="QA prompt template to use from config.yaml. Options: default, detailed, concise")
parser.add_argument("--condense-prompt", default="default", 
                    help="Condense prompt template to use from config.yaml. Options: default, comprehensive, simple")
parser.add_argument("--config", default="config.yaml", 
                    help="Path to configuration file")

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = argparse.Namespace(
        llm="gpt4o-mini",
        embedding="large", 
        qa_prompt="default", 
        condense_prompt="default",
        config="config.yaml"
    )

project_version = f"{args.llm}_{args.embedding}_{args.qa_prompt}_{args.condense_prompt}"
print(f"Project version: {project_version}")

# ============================================================================
# FUTUREAGI INSTRUMENTATION SETUP
# ============================================================================
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType, EvalConfig, EvalName, EvalSpanKind, EvalTag, EvalTagType, ModelChoices

eval_tags = [
    EvalTag(
        eval_name=EvalName.FACTUAL_ACCURACY,
        type=EvalTagType.OBSERVATION_SPAN,
        value=EvalSpanKind.LLM,
        mapping={
            "input": "llm.input_messages.0.message.content",
            "output": "llm.output_messages.0.message.content",
            "context": "llm.input_messages.1.message.content"
        },
        custom_eval_name="factual_accuracy",
        model=ModelChoices.TURING_LARGE
    )
]

trace_provider = register(
    project_type=ProjectType.EXPERIMENT,   
    project_name="REGRESSION-TESTING-RAG",
    project_version_name=project_version,
    eval_tags=eval_tags
)

from traceai_llamaindex import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument(tracer_provider=trace_provider)

# ============================================================================
# CONFIGURATION
# ============================================================================
STORAGE_PATH = Path("./vectorstore")
DOCUMENTS_PATH = Path("./documents")
DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(args.config)
CONFIG = {}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as file:
        CONFIG = yaml.safe_load(file)
else:
    print(f"Warning: Configuration file {CONFIG_PATH} not found, using defaults")

def get_llm_config():
    if not CONFIG or 'llm_models' not in CONFIG or args.llm not in CONFIG['llm_models']:
        print(f"Warning: LLM model '{args.llm}' not found in config, using environment variable or default")
        return {
            'name': os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            'temperature': 0.2
        }
    return CONFIG['llm_models'][args.llm]

def get_embedding_config():
    if not CONFIG or 'embedding_models' not in CONFIG or args.embedding not in CONFIG['embedding_models']:
        print(f"Warning: Embedding model '{args.embedding}' not found in config, using environment variable or default")
        return {
            'name': os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        }
    return CONFIG['embedding_models'][args.embedding]

def create_qa_prompt():
    if not CONFIG or 'qa_prompts' not in CONFIG or args.qa_prompt not in CONFIG['qa_prompts']:
        print(f"Warning: QA prompt '{args.qa_prompt}' not found in config, using default")
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about documents."),
            ("user", "Question: {query_str}\nContext: {context_str}")
        ])
    
    prompt_config = CONFIG['qa_prompts'][args.qa_prompt]
    return ChatPromptTemplate.from_messages([
        ("system", prompt_config['system']),
        ("user", prompt_config['user'])
    ])

def create_condense_prompt():
    if not CONFIG or 'condense_prompts' not in CONFIG or args.condense_prompt not in CONFIG['condense_prompts']:
        print(f"Warning: Condense prompt '{args.condense_prompt}' not found in config, using default")
        return ChatPromptTemplate.from_messages([
            ("system", "Given the following conversation history and a new question, rephrase the question to be a standalone question that captures all relevant context from the conversation."),
            ("user", "Chat History:\n{chat_history}\n\nNew Question: {query_str}\n\nStandalone Question:")
        ])
    
    prompt_config = CONFIG['condense_prompts'][args.condense_prompt]
    return ChatPromptTemplate.from_messages([
        ("system", prompt_config['system']),
        ("user", prompt_config['user'])
    ])

LLM_CONFIG = get_llm_config()
EMBED_CONFIG = get_embedding_config()
QA_PROMPT = create_qa_prompt()
CONDENSE_PROMPT = create_condense_prompt()

DEFAULT_LLM_MODEL = LLM_CONFIG['name']
DEFAULT_EMBED_MODEL = EMBED_CONFIG['name']
DEFAULT_TEMPERATURE = LLM_CONFIG.get('temperature', 0.2)

INDEX: VectorStoreIndex | None = None

# ============================================================================
# INDEX MANAGEMENT
# ============================================================================
def _ensure_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in .env or export it before running."
        )


def _configure_settings(temperature: float = None):
    _ensure_api_key()
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    Settings.llm = OpenAI(model=DEFAULT_LLM_MODEL, temperature=temp)
    Settings.embed_model = OpenAIEmbedding(model=DEFAULT_EMBED_MODEL)


def initialize_index() -> VectorStoreIndex:
    _configure_settings()

    if not any(DOCUMENTS_PATH.iterdir()):
        (DOCUMENTS_PATH / "README.txt").write_text(
            "Add PDFs/TXT/DOCX/MD into ./documents and click 'Rebuild Index'.\n"
        )

    if not STORAGE_PATH.exists():
        docs = SimpleDirectoryReader(str(DOCUMENTS_PATH), recursive=True).load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=str(STORAGE_PATH))
        return index

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_PATH))
    index = load_index_from_storage(storage_context)
    return index


def rebuild_index() -> str:
    global INDEX
    try:
        _configure_settings()
        if STORAGE_PATH.exists():
            for p in STORAGE_PATH.glob("**/*"):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass
            for p in sorted(STORAGE_PATH.glob("**/*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            if STORAGE_PATH.exists():
                STORAGE_PATH.rmdir()
        INDEX = None
        INDEX = initialize_index()
        return "Index rebuilt from ./documents and saved to ./vectorstore."
    except Exception as e:
        return f"Rebuild failed: {type(e).__name__}: {e}"

# ============================================================================
# CHAT ENGINE AND RESPONSES
# ============================================================================
def _history_to_memory(history: List, token_limit: int = 4000) -> ChatMemoryBuffer:
    mem = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
    for msg in history:
        if isinstance(msg, tuple) and len(msg) >= 2:
            user_msg, bot_msg = msg[0], msg[1]
            if user_msg:
                mem.put(ChatMessage(role=MessageRole.USER, content=user_msg))
            if bot_msg:
                mem.put(ChatMessage(role=MessageRole.ASSISTANT, content=bot_msg))
    return mem


def respond(message: str, history: List[Any]) -> str:
    global INDEX
    try:
        if INDEX is None:
            INDEX = initialize_index()

        memory = _history_to_memory(history)
        engine = INDEX.as_chat_engine(
            chat_mode="condense_question",
            memory=memory,
            verbose=True,
            text_qa_template=QA_PROMPT,
            condense_question_template=CONDENSE_PROMPT,
        )
        response = engine.chat(message)

        sources_lines = []
        try:
            for sn in (response.source_nodes or [])[:3]:
                meta = sn.metadata or {}
                name = meta.get("file_name") or meta.get("filename") or meta.get("source") or sn.node_id[:8]
                score = getattr(sn, "score", None)
                page_num = meta.get("page_label") or meta.get("page") or meta.get("page_number")
                
                source_text = f"- {name}"
                if page_num is not None:
                    source_text += f" (page {page_num})"
                if score is not None:
                    source_text += f" (score={round(score, 3)})"
                
                sources_lines.append(source_text)
        except Exception as e:
            print(f"Error formatting sources: {str(e)}")
            pass

        if sources_lines:
            return f"{response.response}\n\n**Sources**\n" + "\n".join(sources_lines)
        else:
            return response.response

    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"

# ============================================================================
# FILE HANDLING
# ============================================================================
def save_uploaded(files) -> str:
    if not files:
        return "No files uploaded."
    saved = 0
    for f in files:
        try:
            dest = DOCUMENTS_PATH / Path(f.name).name
            with open(f.name, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
            saved += 1
        except Exception:
            pass
    return f"Uploaded {saved} file(s) to ./documents. Click 'Rebuild Index' to include them."

# ============================================================================
# UI CONFIGURATION
# ============================================================================
DESCRIPTION = f"""
Upload documents and chat with their content using AI.

**Current Configuration:**
- LLM: {args.llm} ({DEFAULT_LLM_MODEL})
- Embedding: {args.embedding} ({DEFAULT_EMBED_MODEL})
- QA Prompt: {args.qa_prompt}
- Condense Prompt: {args.condense_prompt}
- Temperature: {DEFAULT_TEMPERATURE}
- Version ID: {project_version}
"""

CSS = """
.container {border-radius: 10px; padding: 20px; margin-bottom: 10px}
.status-container {min-height: 30px; margin-top: 10px}
"""

def upload_and_process(files):
    upload_result = save_uploaded(files)
    if files and "Uploaded" in upload_result:
        index_result = rebuild_index()
        return "Documents uploaded successfully. Start chatting."
    return upload_result

# ============================================================================
# UI LAYOUT AND INITIALIZATION
# ============================================================================
with gr.Blocks(title="Document Chat Assistant", fill_height=True, css=CSS) as demo:
    gr.Markdown("#Document Chat Assistant")
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="container"):
            files = gr.Files(label="Upload Documents (PDF, TXT, DOCX, MD)", file_count="multiple")
            upload_btn = gr.Button("Upload", variant="primary")
            status = gr.Markdown("", elem_classes="status-container")
        
        with gr.Column(scale=2, elem_classes="container"):
            chat = gr.ChatInterface(
                fn=respond,
                textbox=gr.Textbox(placeholder="Ask a question about your documents..."),
                examples=[
                    "Summarize the key points from the document.",
                    "What are the main concepts discussed in the document?",
                    "Extract the most important conclusions from the document.",
                    "Compare and contrast the main ideas in the document.",
                ],
            )

    upload_btn.click(upload_and_process, inputs=[files], outputs=[status])

# ============================================================================
# MAIN FUNCTION
# ============================================================================
if __name__ == "__main__":
    print(f"\nRunning with configuration:")
    print(f"- LLM: {args.llm} ({DEFAULT_LLM_MODEL})")
    print(f"- Embedding: {args.embedding} ({DEFAULT_EMBED_MODEL})")
    print(f"- QA Prompt: {args.qa_prompt}")
    print(f"- Condense Prompt: {args.condense_prompt}")
    print(f"- Temperature: {DEFAULT_TEMPERATURE}")
    print(f"- Config File: {args.config}\n")
    
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_api=False)