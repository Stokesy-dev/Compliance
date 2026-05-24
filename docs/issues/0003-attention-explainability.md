# Issue #3: Token-Level Self-Attention Explainability Layer

## What to build
Develop the explainability pipeline in `explainability/lime_analysis.py` to highlight which words drove the risk classification.

The system must:
1. Extract the raw token-level self-attention weights from the final layer of our classification pipeline (either BART or DistilBERT) during the single inference forward pass.
2. Filter and aggregate attention scores corresponding to the classification token (`[CLS]`).
3. Normalize the attention scores across all non-special tokens, generating a relative importance mapping (0.0 to 1.0) for every word in the document.
4. Update `main.py --mode classify` to output a token-attention keyword importance report showing which tokens highly influenced the classification category (e.g., words with attention > 0.5 highlighted).
5. Verify with unit tests that attention weights are returned and correctly sum/normalize across tokens.

## Status
Completed

## Acceptance criteria
- [x] Classification pipeline outputs self-attention weights in addition to probabilities.
- [x] Attention weights are aggregated and normalized into relative token-level scores (0.0 to 1.0) in `explainability/lime_analysis.py`.
- [x] `main.py --mode classify` prints the token-attention report.
- [x] `pytest` validates that the extraction runs in sub-second time during a single forward pass and outputs valid scores.

## Blocked by
- [Issue #1: Project Bootstrapping & Local Zero-Shot Classifier CLI](file:///Users/sohamwarad/compliance/docs/issues/0001-project-bootstrap-and-zero-shot-cli.md)
