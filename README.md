# Examples

This repository contains example usecases integrating Future AGI and TraceAI SDK.

## 1. `Computer_UseAgent`

This example demonstrates an autonomous AI agent that can browse and interact with a computer system. Using the TraceAI library for instrumentation, the agent can navigate through files, execute commands, and perform various computer operations while maintaining detailed logs of its actions and decision-making process.

## 2. `ecom_agent`

This example demonstrates a Gradio-based e-commerce agent that uses TraceAI for instrumentation and evaluation. The agent provides an interactive interface for handling user queries with images, product searches, order management, and product recommendations with image rendering capabilities.

## 3. `external_gdrive_search`

This example shows a LlamaIndex-based demo application for searching across an internal knowledge database and Google Drive files using semantic search with Retrieval-Augmented Generation (RAG).

## 4. `font_generator`

This example demonstrates a Gradio-based font generator that creates custom font samples using OpenAI's 4o image generation model. Users can describe their desired font style, and the application will generate 5 distinct variations of the font, each optimized for different use cases (base, bold, italic, decorative, and minimalist styles).

## 5. `font_search`

This example shows a Gradio-based font search application that uses TraceAI for instrumentation and evaluation. The application allows users to search for fonts based on their description, and the application will return a list of fonts that match the user's description.

## 6. `interview_agent`
This example demonstrates an interview agent that transcribes audio interviews, generates summaries, stores them in a vector database, and enables semantic search and analysis across multiple interviews using OpenAI's models, ChromaDB, and TraceAI for instrumentation and evaluation.

## 7. `multi_agent`

This example demonstrates a multi-agent orchestration system that leverages LangGraph and LangChain to create a network of specialized agents managed by a supervisor. The system can perform tasks like information retrieval and document creation with minimal human intervention.

## 8. `rag-agent`
This example demonstrates a Retrieval-Augmented Generation (RAG) system that enhances question-answering capabilities by breaking down complex questions into sub-questions, retrieving relevant documents, and generating comprehensive answers using OpenAI's language models. The project leverages LangChain, OpenTelemetry, and Chroma for document retrieval and semantic chunking.

## 9. `qdrant-rag-decay-workshop`

This example is the full "Why Did My RAG Agent Get Worse? — A Live Autopsy" workshop (Future AGI x Qdrant): an agentic Pokedex RAG system that decays as its corpus grows from 151 to 1,025 species, then recovers through four measured fixes — dedup, embedding migration via named vectors, hybrid retrieval with ColBERT reranking, and an `is_current` payload filter that closes a stale-document failure no ranking fix can touch. Every stage's 37-query golden run is traced to its own Future AGI project and scored with five trace evals plus a ground-truth answer-correctness judge; answer correctness climbs 0.57 → 0.92 across the arc. Includes the Streamlit app with a retrieval panel, the live-fix notebook, the webinar runbook, and committed run data so every quoted number reproduces.
