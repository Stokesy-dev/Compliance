# Issue #6: Premium Streamlit Multi-Tab Dashboard

## What to build
Build a visually stunning, highly interactive Streamlit dashboard at `dashboard/streamlit_app.py` leveraging modern web design aesthetics (custom fonts, curated HSL color schemes, glassmorphism, dynamic micro-animations, and highlighted text overlays).

The dashboard must feature 4 fully functional tabs:
1. **Tab 1 — Document Upload & Classification**: Enables file uploads, displays predicted risk category with dynamic gauge/confidence scores, and shows our visual **Classification Explanation** (attention weight highlights overlaying the text with color gradients).
2. **Tab 2 — Entity Extraction Explorer**: Highlights extracted organizations, dates, and monetary values in the text body using spaCy interactive entity visualizations.
3. **Tab 3 — RAG Query Chat**: A premium chat container allowing natural language questions across the uploaded document corpus with instant response delivery.
4. **Tab 4 — Model Performance Metrics**: Visualizes trainer metrics including F1-score progress, training loss, and an interactive confusion matrix for compliance officers to audit the model's metrics.

## Acceptance criteria
- [ ] Aesthetic interface uses premium HSL colors, modern typography (e.g. Outfit/Inter), and contains zero plain black/white or default Streamlit styling.
- [ ] Uploading a file updates Tab 1, Tab 2, and indexes the document into the RAG vector store for Tab 3 instantly.
- [ ] Explainability highlights (Tab 1) and spaCy entities (Tab 2) render correctly.
- [ ] RAG Chat bot responds gracefully (supporting both Groq API and the local fallback generator).
- [ ] Running `streamlit run dashboard/streamlit_app.py` executes cleanly without warning banners.

## Blocked by
- [Issue #2: spaCy Named Entity Recognition Preprocessing Pipeline](file:///Users/sohamwarad/compliance/docs/issues/0002-spacy-ner-pipeline.md)
- [Issue #3: Token-Level Self-Attention Explainability Layer](file:///Users/sohamwarad/compliance/docs/issues/0003-attention-explainability.md)
- [Issue #4: Persistent FAISS Vector Store & RAG Pipeline with Offline Fallback](file:///Users/sohamwarad/compliance/docs/issues/0004-faiss-rag-pipeline.md)
- [Issue #5: Offline Classifier Fine-Tuning Pipeline](file:///Users/sohamwarad/compliance/docs/issues/0005-classifier-training-pipeline.md)
