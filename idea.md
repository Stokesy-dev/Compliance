Project Brief
Project Name: Regulatory Compliance NLP Classifier & GenAI Report Summarizer
Problem Statement:
Compliance teams at financial institutions manually review thousands of regulatory documents daily to categorize risk and extract actionable insights — a slow, error-prone process. Build an intelligent system that automatically classifies compliance documents by risk category and enables natural language querying over document corpora using Generative AI.

Technical Specification
Data:

LexGLUE benchmark dataset from HuggingFace — publicly available legal/regulatory text corpus
Use ECtHR or SCOTUS subset for multi-class classification
Documents cover financial crime, fraud, regulatory breach categories

Classification Layer:

Fine-tune DistilBERT using HuggingFace Transformers on 10K+ regulatory documents
Multi-class classification — financial crime, fraud, regulatory breach
Evaluate using F1 score, precision, recall on held-out test set
Target 84% F1 on test set

NLP Preprocessing Layer:

spaCy pipeline for named entity recognition — extract dates, organizations, monetary values
Pandas for data cleaning, deduplication, and structuring
Tokenization, stopword removal, text normalization

RAG Pipeline:

sentence-transformers (all-MiniLM-L6-v2) for document chunk embeddings
FAISS vector store for semantic indexing of document corpus
LangChain for orchestrating retrieval and generation
Groq's Llama 3.1 as the generation model for plain-language summaries and Q&A
Sub-second retrieval latency target

Explainability:

LIME or attention visualization on DistilBERT classifications
Show which tokens drove the risk classification decision

Dashboard (Streamlit):

Tab 1 — Document upload and classification (shows risk category + confidence score)
Tab 2 — Entity extraction view (dates, orgs, monetary values highlighted)
Tab 3 — RAG query interface (natural language Q&A over uploaded documents)
Tab 4 — Model performance metrics (F1, precision, recall, confusion matrix)

CLI:

python main.py --mode classify --input docs/sample.pdf
python main.py --mode query --question "What are the fraud risk factors?"

Testing:

pytest suite covering preprocessing, classification, retrieval, and generation pipeline components


Tech Stack
Python, Pandas, Scikit-learn, HuggingFace Transformers (DistilBERT), sentence-transformers, FAISS, LangChain, Groq API, Llama 3.1, spaCy, Streamlit, SQL, pytest, LIME

File Structure
compliance-nlp-classifier/
├── data/
│   └── load_dataset.py
├── models/
│   ├── classifier.py
│   └── rag_pipeline.py
├── preprocessing/
│   └── nlp_pipeline.py
├── explainability/
│   └── lime_analysis.py
├── dashboard/
│   └── streamlit_app.py
├── main.py
├── tests/
│   └── test_pipeline.py
└── requirements.txt

Expected Outcomes

84% F1 score on multi-class compliance risk classification
Sub-second semantic retrieval latency on FAISS index
Named entity extraction across dates, organizations, monetary values
Natural language Q&A over regulatory document corpus via RAG
End-to-end pytest validated pipeline
Fully interactive Streamlit interface for non-technical compliance officers


Resume Bullets to Justify

Fine-tuned a DistilBERT text classifier on 10K+ regulatory documents from the LexGLUE benchmark, achieving 84% F1 on multi-class compliance risk categorization (financial crime, fraud, regulatory breach).
Engineered a RAG pipeline using FAISS vector search and Groq's Llama 3.1, enabling natural-language querying of large unstructured compliance corpora with sub-second retrieval latency.
Developed an automated data preprocessing pipeline using spaCy and Pandas to extract entities (dates, organizations, monetary values) from raw legal text, reducing manual annotation time and improving downstream model accuracy by 11%.
Deployed a comprehensive Streamlit interface with upload, classify, and query workflows, empowering non-technical compliance officers to interact with AI-generated document summaries and risk flags without writing code.


Interview Angles
Why this project matters:

"Wolters Kluwer's FCC division processes thousands of compliance documents manually. I built a system that automates the two most time-consuming tasks — risk categorization and document querying — using a fine-tuned BERT classifier and a RAG pipeline. The FAISS-backed retrieval means analysts get answers in under a second instead of manually searching through hundreds of pages."

Why DistilBERT over full BERT:

"DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its performance — for a production compliance tool where speed matters, that tradeoff makes sense."

Why RAG over pure LLM:

"A pure LLM hallucinates on domain-specific compliance content. RAG grounds the generation in actual retrieved document chunks, which is critical when the output is being used for regulatory decision-making."