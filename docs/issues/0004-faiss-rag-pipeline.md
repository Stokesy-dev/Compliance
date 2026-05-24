# Issue #4: Persistent FAISS Vector Store & RAG Pipeline with Offline Fallback

## What to build
Build the core RAG pipeline under `models/rag_pipeline.py` using `sentence-transformers/all-MiniLM-L6-v2` for chunk embeddings and `FAISS` for semantic indexing.

The system must:
1. Handle multi-page PDF/text parsing and split them into chunks using a LangChain character splitter.
2. Build or load a persistent **FAISS Vector Store** on disk inside `data/faiss_index/`. When new documents are uploaded, it chunks, embeds, and appends them to this session index.
3. Implement the Q&A generator using LangChain with Groq's Llama 3.1.
4. Implement a robust **RAG Generation Fallback**: If the `GROQ_API_KEY` is not present, it retrieves the top 3 relevant chunks, extracts them, and generates a structured, beautiful local markdown report summarizing the answers offline, avoiding app crashes.
5. Add CLI flag `main.py --mode query --question "<question_text>"` to search the persistent index and print the answer.
6. Verify with `pytest` unit tests for FAISS persistence, context retrieval, and successful execution of the offline fallback report.

## Acceptance criteria
- [ ] Text chunking and `sentence-transformers` embeddings integrated.
- [ ] FAISS index is saved/loaded from disk inside `data/faiss_index/`.
- [ ] Running query via CLI connects to Groq API or triggers the local markdown synthesizer fallback cleanly without raising exceptions.
- [ ] Sub-second retrieval latency achieved for index lookup.
- [ ] `pytest` validates indexing, persistence, and local synthesizer fallback.

## Blocked by
- [Issue #1: Project Bootstrapping & Local Zero-Shot Classifier CLI](file:///Users/sohamwarad/compliance/docs/issues/0001-project-bootstrap-and-zero-shot-cli.md)
