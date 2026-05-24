# Issue #5: Offline Classifier Fine-Tuning Pipeline

## What to build
Build the complete dataset loading, mapping, and model fine-tuning pipeline to replace the zero-shot classifier with our high-accuracy custom model.

The system must:
1. Sourced in `data/load_dataset.py`, download the LexGLUE SCOTUS dataset and apply our deterministic mapping (detailed in `ADR 0001`) to map SCOTUS issues to **Financial Crime**, **Fraud**, and **Regulatory Breach** labels, filtering out irrelevant records.
2. In `models/classifier.py`, construct a standard Hugging Face `Trainer` to fine-tune `distilbert-base-uncased` on these mapped training samples (minimum 10K records).
3. Save checkpoints inside `models/checkpoints/best_model/`.
4. Support the training mode command `python main.py --mode train`.
5. Ensure the inference engine updates to automatically load the custom **Model Checkpoint** from `models/checkpoints/best_model/` rather than the zero-shot pipeline when present.
6. Write unit tests confirming that `load_dataset` downloads, maps, and structures inputs correctly.

## Status
Completed

## Acceptance criteria
- [x] LexGLUE dataset downloads and filters cleanly to our three custom classes.
- [x] Running `python main.py --mode train` executes the fine-tuning trainer loop and outputs a checkpoint model in `models/checkpoints/best_model/`.
- [x] Subsequent runs of `main.py --mode classify` seamlessly load the newly created custom DistilBERT model.
- [x] `pytest` validates data mappings and training loop initiation on a tiny dummy subset.

## Blocked by
- [Issue #1: Project Bootstrapping & Local Zero-Shot Classifier CLI](file:///Users/sohamwarad/compliance/docs/issues/0001-project-bootstrap-and-zero-shot-cli.md)
