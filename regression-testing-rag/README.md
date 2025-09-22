# RAG Regression Testing Framework

A configurable document chat application with built-in regression testing capabilities for RAG (Retrieval-Augmented Generation) systems. This tool helps you evaluate and compare the performance of different LLM models, embedding models, and prompt templates in a RAG context.

## Features

- **Document Chat**: Upload and chat with documents (PDF, TXT, DOCX, MD)
- **Configurable Components**:
  - LLM models (GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5)
  - Embedding models (text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002)
  - QA prompt templates
  - Chat history condensation prompt templates
- **Evaluations**: Integration with FutureAGI instrumentation for evaluating:
  - Factual accuracy
  - Context relevance
  - Task completion
  - Hallucination detection
  - Chunk attribution
  - Chunk utilization
- **Interactive UI**: Gradio-based interface for easy document upload and chatting

## Requirements

- Python 3.8+
- OpenAI API key
- FutureAGI API key and Secret key (get your keys from [FutureAGI dashboard](https://app.futureagi.com/dashboard/keys))

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:future-agi/cookbooks.git
   cd cookbooks/cookbooks/regression-testing-rag
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:

   ```
   # OpenAI keys
   OPENAI_API_KEY=your-api-key-here

   # FutureAGI keys
   FI_API_KEY=your-futureagi-api-key
   FI_SECRET_KEY=your-futureagi-secret-key
   ```

## Configuration

The system is configured through the `config.yaml` file, which allows you to define:

- LLM models and their parameters
- Embedding models
- QA prompt templates
- Condense prompt templates for chat history

Example configuration:

```yaml
# LLM models configuration
llm_models:
  gpt4o-mini:
    name: gpt-4o-mini
    temperature: 0.2
  
  gpt4o:
    name: gpt-4o
    temperature: 0.2

# Embedding models configuration
embedding_models:
  large:
    name: text-embedding-3-large
  
  small:
    name: text-embedding-3-small

# Prompt templates
qa_prompts:
  default:
    system: "You are a helpful assistant that answers questions about documents."
    user: "Question: {query_str}\nContext: {context_str}"
```

## Usage

1. Run the application:

   ```bash
   python app.py --llm gpt4o-mini --embedding large --qa-prompt default --condense-prompt default
   ```
2. Access the UI at `http://localhost:7860`
3. Upload documents and start chatting!

### Command Line Arguments

- `--llm`: LLM model to use (default: gpt4o-mini)
- `--embedding`: Embedding model to use (default: large)
- `--qa-prompt`: QA prompt template (default: default)
- `--condense-prompt`: Condense prompt template (default: default)
- `--config`: Path to configuration file (default: config.yaml)

## Project Structure

- `app.py`: Main application code
- `config.yaml`: Configuration for models and prompts
- `requirements.txt`: Python dependencies
- `documents/`: Directory where uploaded documents are stored
- `vectorstore/`: Directory where the vector index is saved

## Evaluation Metrics

The system includes automatic evaluation using FutureAGI instrumentation with the following metrics:

1. **Factual Accuracy**: Evaluates if responses are factually correct based on the document content
2. **Context Relevance**: Measures how relevant the retrieved context is to the user query
3. **Task Completion**: Assesses if the response successfully completes the user's requested task
4. **Hallucination Detection**: Identifies statements not supported by the provided context
5. **Chunk Attribution**: Tracks which parts of the context are used in the response
6. **Chunk Utilization**: Measures how effectively the system uses the retrieved chunks

## License

[License information]
