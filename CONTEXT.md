# Domain Glossary

This document serves as the single source of truth for the domain language and terminology used across the Regulatory Compliance NLP Classifier & GenAI Report Summarizer system.

---

## Terms

### Document
An unstructured regulatory or compliance text artifact (typically a PDF or text file) uploaded by a compliance officer for active analysis, risk classification, entity extraction, and semantic querying.

### Training Sample
A pre-labeled, static legal text record sourced from the LexGLUE benchmark corpus. Training samples are used exclusively offline to fine-tune the classification model and evaluate its performance. They are never ingested into the runtime RAG pipeline.

### Document Corpus
The collection of active Documents uploaded by the user that are indexed and available for semantic search and question-answering during a session.

### Financial Crime
A compliance risk category involving illegal acts committed by individuals or corporations to obtain a financial or professional advantage, including banking violations, securities fraud, tax evasion, and antitrust issues.

### Fraud
A compliance risk category involving intentional deception or misrepresentation to secure unfair or unlawful gain, such as bribery, theft, or fraudulent misstatements.

### Regulatory Breach
A compliance risk category representing violations of rules, directives, or standards set by administrative and regulatory authorities (e.g., SEC, FTC, IRS rules) and general non-compliance with statutory obligations.

### Entity
A specific piece of structured information (specifically Dates, Organizations, and Monetary Values) identified and extracted from a Document using the spaCy Named Entity Recognition (NER) pipeline.

### Classification Explanation
The token-level self-attention weights extracted from the final layer of the fine-tuned DistilBERT model. This represents the relative importance of each word/token in driving the model's risk classification decision.

### FAISS Vector Store
The semantic index constructed from dense embeddings of text chunks from all uploaded Documents. It allows sub-second retrieval of relevant passages based on cosine similarity of natural language questions.

### RAG Generation Fallback
A local generation mechanism that triggers when external LLM APIs (e.g., Groq) are unavailable. It provides structured summaries and context extraction directly from retrieved chunks without requiring internet access or API credentials.

### Model Checkpoint
A serialized set of weights and configurations saved from a fine-tuning run of the DistilBERT classifier. The primary checkpoint is stored at `models/checkpoints/best_model/` and is loaded by default for high-accuracy classification.

### Zero-Shot Fallback
An alternative classification pipeline loaded at runtime when a fine-tuned Model Checkpoint is not yet trained or available. It uses a pre-trained zero-shot model (e.g., BART-large-MNLI) to categorize risk categories out-of-the-box.




