# Issue #1: Project Bootstrapping & Local Zero-Shot Classifier CLI

## What to build
Set up the workspace configuration, core directories, and requirements. Implement the unified CLI in `main.py` supporting `--mode classify --input <path>`. 

The system must:
1. Gracefully check for the custom fine-tuned **Model Checkpoint** in `models/checkpoints/best_model/`.
2. If missing, automatically activate the **Zero-Shot Fallback** using `facebook/bart-large-mnli` to classify the input text.
3. Output the predicted compliance risk category (**Financial Crime**, **Fraud**, or **Regulatory Breach**) along with the model's confidence score to the console.
4. Set up an initial test framework using `pytest` to verify this zero-shot path works out-of-the-box.

## Acceptance criteria
- [ ] Core directory structure created (`data/`, `models/`, `preprocessing/`, `explainability/`, `dashboard/`, `tests/`).
- [ ] `requirements.txt` includes `transformers`, `torch`, `scikit-learn`, `pytest`, `pandas`, `numpy`.
- [ ] Running `python main.py --mode classify --input path/to/text.txt` loads the Zero-Shot fallback, successfully runs classification, and prints the predicted category and confidence.
- [ ] Baseline test `tests/test_pipeline.py` verifies that zero-shot classification maps raw text to the three target categories cleanly.
- [ ] Code complies with all defined domain glossary terms in `CONTEXT.md` and respects `ADR 0004`.

## Blocked by
None - can start immediately
