# Issue #2: spaCy Named Entity Recognition Preprocessing Pipeline

## What to build
Develop the NLP Preprocessing module in `preprocessing/nlp_pipeline.py` using spaCy (`en_core_web_sm`) to clean text, handle tokenization, and perform Named Entity Recognition (NER).

The system must:
1. Extract named **Entities** specifically targeting **Dates**, **Organizations**, and **Monetary Values**.
2. Integrate this preprocessing step into our existing classification pipeline.
3. Update `main.py --mode classify` so that it runs spaCy NER in parallel with classification, printing all extracted entities (text, entity label, and character span indices) in the terminal output.
4. Add comprehensive `pytest` unit tests verifying that standard sample texts containing dates, cash amounts, and corporate names are cleaned and their entities are correctly identified with correct boundary spans.

## Status
Completed

## Acceptance criteria
- [x] `spacy` library and `en_core_web_sm` model integrated.
- [x] Preprocessing functions clean, normalize, and extract entities correctly.
- [x] `main.py --mode classify` outputs the risk classification *and* a formatted list of all extracted entities.
- [x] `tests/test_pipeline.py` contains automated assertions for spaCy cleaning and entity detection boundaries.

## Blocked by
- [Issue #1: Project Bootstrapping & Local Zero-Shot Classifier CLI](file:///Users/sohamwarad/compliance/docs/issues/0001-project-bootstrap-and-zero-shot-cli.md)
