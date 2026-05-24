# ADR 0003: Vector Store Persistence and LLM Fallback Strategy

## Status
Proposed (Pending approval in current plan)

## Context
The RAG pipeline requires:
1. Embedding and indexing uploaded compliance documents using `sentence-transformers` and `FAISS`.
2. Sending retrieved contexts to an LLM (Groq's Llama 3.1) to generate plain-language answers and summaries.

We need to decide:
1. How and where the vector index is stored.
2. What happens if the `GROQ_API_KEY` is missing or the external API is unreachable, especially in offline test suites and developer environments.

## Decision
1. **Vector Store Lifecycle**: We will build a unified vector store in `data/faiss_index` that persists on disk during a dashboard session. The Streamlit app will load this index on start, append new document chunks dynamically when the user uploads a document, and persist the updated index. This allows the user to query both the newly uploaded document and historical documents in their corpus.
2. **LLM Fallback Strategy**: If the `GROQ_API_KEY` environment variable is not defined or is empty, the RAG pipeline will automatically activate a local **RAG Generation Fallback**. This fallback will extract the highly relevant retrieved chunks from the vector store, compile them, and format them as a structured markdown report summarizing the key findings. This guarantees that the system remains fully operational and testable offline, while notifying the user in the UI that the Groq LLM API is currently offline.

## Alternatives Considered
* **Pure In-Memory Vector Store**: Rejected because users would lose all indexed documents every time the Streamlit dashboard or CLI is restarted, which degrades the user experience.
* **Failing on Missing API Key**: Rejected as it makes testing, CI/CD pipelines, and local development highly fragile and dependent on external API keys.

## Consequences
* **Positive**: Fast, consistent local testing and seamless user experience across app restarts.
* **Positive**: Resilience to network issues and API rate limits.
* **Negative**: The local fallback report does not have the conversational intelligence of Llama 3.1, but it is highly informative as it synthesizes the exact retrieved compliance passages.
