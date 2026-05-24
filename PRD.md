# Product Requirement Document (PRD)

## Problem Statement
Compliance teams at financial institutions are forced to manually review thousands of complex, unstructured regulatory and legal documents daily. This manual categorization of risk categories and parsing of critical entities (e.g., dates, names, amounts) is a slow, tedious, and highly error-prone process. Furthermore, compliance officers lack an intelligent, natural-language way to query their active document corpora, forcing them to manually search across hundred-page files to find specific compliance details or potential violations.

---

## Solution
An intelligent, high-performance web application and CLI system—the **Regulatory Compliance NLP Classifier & GenAI Report Summarizer**—that automates:
1. **Multi-Class Classification**: Automatic categorization of documents into core compliance risk categories (Financial Crime, Fraud, Regulatory Breach).
2. **Explainable AI**: Visual token-level attention mapping showing exactly which words triggered specific risk classifications.
3. **Named Entity Recognition (NER)**: Sub-second extraction and visual highlighting of critical dates, organizations, and monetary values.
4. **Conversational Retrieval-Augmented Generation (RAG)**: Fast semantic Q&A across the uploaded document corpus utilizing local FAISS vector search and Llama 3.1, supported by a resilient local synthesizer fallback when offline.

---

## User Stories

1. **As a compliance officer**, I want to upload a multi-page PDF or text document, so that the system can automatically classify its compliance risk profile.
2. **As a compliance officer**, I want to see the specific category of compliance risk (Financial Crime, Fraud, Regulatory Breach) along with a model confidence score, so that I can prioritize high-risk documents.
3. **As a compliance analyst**, I want to view a visual explanation showing exactly which words in the document influenced the model's classification, so that I can audit the AI's reasoning.
4. **As a regulatory inspector**, I want to view extracted named entities (dates, organizations, monetary values) highlighted within the document, so that I don't have to manually scan pages for critical metadata.
5. **As a compliance officer**, I want to ask natural language questions about all the documents I have uploaded in the session, so that I can quickly extract key dates, clauses, or liabilities without reading the files in full.
6. **As a system administrator**, I want the Q&A RAG pipeline to have sub-second retrieval latency, so that compliance queries are resolved without making users wait.
7. **As an offline developer or tester**, I want the system to remain fully operational and continue to provide structured RAG summaries even if the external Groq LLM API is unavailable, so that the application is resilient and testable offline.
8. **As a new developer setup engineer**, I want to run the application immediately after cloning without being blocked by a long model training run, so that I can verify full application functionality instantly.
9. **As a data scientist**, I want to run an automated, repeatable script to fine-tune the classification model on labeled regulatory datasets, so that I can continuously improve and validate classifier performance.
10. **As a developer**, I want a unified CLI tool to classify documents and run queries directly from the terminal, so that I can integrate this system into automated cron jobs and server tasks.

---

## Implementation Decisions

### 1. Unified Domain Vocabularies & State
All modules must respect the domain concepts defined in the Domain Glossary:
* **Document**: A user-submitted PDF or text file under review.
* **Training Sample**: An offline record from the LexGLUE SCOTUS corpus.
* **Document Corpus**: The active collection of user-uploaded files.
* **Financial Crime / Fraud / Regulatory Breach**: The deterministic mapped risk categories.
* **Entity**: Extracted spaCy NER data (Dates, Organizations, Monetary Values).
* **Classification Explanation**: Token importance scores computed from DistilBERT final-layer self-attention weights.
* **FAISS Vector Store**: PERSISTENT on-disk vector database storing active session embeddings.
* **RAG Generation Fallback**: The markdown synthesis report when external APIs are offline.
* **Model Checkpoint**: Saved DistilBERT weights from a fine-tuning run.
* **Zero-Shot Fallback**: Pre-trained BART model used when the checkpoint is unavailable.

### 2. Core Deep Modules

* **Data Mapping Module (`data/load_dataset.py`)**
  * Encapsulates downloading LexGLUE SCOTUS.
  * Encapsulates the deterministic mapping of SCOTUS issue area integer IDs to our three target compliance classes (Financial Crime, Fraud, Regulatory Breach), filtering out irrelevant issue fields to maintain high data quality.

* **NLP Preprocessing & Extraction Module (`preprocessing/nlp_pipeline.py`)**
  * Encapsulates the `spacy` English NER model and standard text clean/normalize functions.
  * Exposes a simple interface to extract entity spans and clean raw texts.

* **Classification & Inference Module (`models/classifier.py`)**
  * Implements the **Hugging Face Trainer** pipeline for offline fine-tuning.
  * Exposes an inference function that checks for the presence of a **Model Checkpoint** in `models/checkpoints/best_model/`. If missing, it dynamically invokes the **Zero-Shot Fallback** using `facebook/bart-large-mnli`.
  * Computes **Classification Explanations** by capturing final self-attention layer weights on the CLS token in a single forward pass, returning normalized token weights.

* **RAG Retrieval & Generation Module (`models/rag_pipeline.py`)**
  * Manages PDF parsing and chunking using LangChain wrappers.
  * Encapsulates creating and loading a **FAISS Vector Store** on-disk in `data/faiss_index`.
  * Orchestrates Groq's Llama 3.1 generator. If `GROQ_API_KEY` is missing or is empty, automatically triggers the **RAG Generation Fallback** to output structured context reports in markdown.

### 3. Interface Layer
* **Streamlit Web Dashboard (`dashboard/streamlit_app.py`)**: A modern, interactive multi-tab design with visual text highlights, live Q&A chatbot, and interactive performance charts.
* **CLI Wrapper (`main.py`)**: Provides clean flags (`--mode train`, `--mode classify`, `--mode query`).

---

## Testing Decisions

### 1. Focus on External Behavior
Tests must strictly focus on validating module inputs and outputs, rather than asserting on internal pipeline states, third-party libraries, or Hugging Face internals. This ensures the tests are robust and don't break when model backends or library versions are updated.

### 2. Tested Modules & Scenarios
* **Preprocessing Module**: Validate that spaCy correctly identifies and extracts dates, monetary symbols/numbers, and organizations from dummy texts.
* **Classifier Inference Module**:
  * Verify that the fallback mechanism switches to Zero-Shot classifier gracefully if no fine-tuned model checkpoint is found.
  * Verify that the attention scoring outputs an importance weight dictionary containing all non-padding tokens.
* **RAG Pipeline Module**:
  * Assert that a FAISS index directory is generated on-disk upon document ingestion.
  * Assert that searching the vector index returns relevant context chunks.
  * Verify that the RAG Generation Fallback functions perfectly without a Groq API key, returning a formatted markdown report.

---

## Out of Scope
* Integration with enterprise Document Management Systems (e.g., SharePoint, Documentum).
* Multi-user authentication, user roles, or tenant-level data segregation.
* Online continuous model training/active learning (training is strictly offline/on-demand).
* Visual parsing of complex image diagrams inside PDFs (OCR is out of scope; standard text extraction is used).

---

## Further Notes
* All model weights (`models/checkpoints/`) and vector store index data (`data/faiss_index/`) must be listed in `.gitignore` to prevent accidental pushing of large binary files to source control.
* Groq models will utilize `llama-3.1-8b-instant` for ultra-fast, cost-effective inference.
