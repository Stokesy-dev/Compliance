# ADR 0004: Dual-Mode Classification and Model Fallback Strategy

## Status
Proposed (Pending approval in current plan)

## Context
Fine-tuning DistilBERT on 10,000+ legal training samples from the LexGLUE SCOTUS dataset requires significant compute power and training time (e.g., 30–60 minutes on local CPUs). 

If the application strictly depends on this custom model checkpoint being present, developers, testers, and CI/CD pipelines cannot run the app immediately upon cloning. This makes local developer iteration extremely slow and fragile.

## Decision
We will design a **dual-mode classification pipeline** in `models/classifier.py`:
1. **Fine-Tuning Mode**: A command-line script (`python models/classifier.py --train`) that executes the end-to-end Hugging Face training loop, downloads the LexGLUE SCOTUS dataset, maps issue codes to our risk categories, trains the model, and saves the weights to `models/checkpoints/best_model/`.
2. **Zero-Shot Fallback Mode**: The Streamlit dashboard and inference CLI will dynamically check for the presence of the `best_model` checkpoint. If found, it loads the custom fine-tuned model. If not found, it gracefully loads a high-quality pre-trained Zero-Shot Classification pipeline (specifically `facebook/bart-large-mnli`) and maps the output labels to our categories.

## Alternatives Considered
* **Enforce Local Training on First Start**: Rejected because it halts execution for up to an hour and could crash systems without sufficient RAM or network bandwidth to download datasets instantly.
* **Ship Checkpoints in Git**: Rejected because deep learning model weights are too large (250MB+) to store in a standard Git repository.

## Consequences
* **Positive**: 100% immediate out-of-the-box working application for anyone cloning the repository.
* **Positive**: Simplified continuous integration (CI) and local testing since tests can execute using the zero-shot pipeline.
* **Negative**: The Zero-Shot fallback has slightly lower domain-specific classification accuracy than the fully fine-tuned DistilBERT model, but it serves as an excellent functional bridge.
