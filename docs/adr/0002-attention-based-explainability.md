# ADR 0002: Attention-Based Explainability Over LIME

## Status
Proposed (Pending approval in current plan)

## Context
Compliance officers require explainability for the risk classification output (i.e., understanding *why* the fine-tuned DistilBERT model flagged a document as "Financial Crime", "Fraud", or "Regulatory Breach"). 

The initial specification proposed using LIME (Local Interpretable Model-agnostic Explanations) or attention visualization. LIME perturbs input documents by masking words and running hundreds of inference passes, which incurs a significant computational and latency penalty (5–15 seconds per document). This makes LIME poorly suited for interactive, sub-second responses inside a Streamlit web application.

## Decision
We will use **DistilBERT's native self-attention weights** from the final self-attention layer to compute token-level importance, rather than LIME. 

The implementation will:
1. Enable `output_attentions=True` during the model forward pass.
2. Extract the attention weights from the final layer.
3. Average the attention scores across all heads for the `[CLS]` token (which is used for the classification prediction) to obtain a relative importance score for each input word.
4. Scale and normalize these scores to map them directly to highlighted spans in the Streamlit UI.

## Alternatives Considered
* **LIME**: Rejected due to high latency overhead and complexity of managing perturbation environments in an interactive dashboard.
* **Integrated Gradients**: Also considered but rejected as it requires computing gradients with respect to inputs, which is more complex to implement and verify compared to direct attention extraction.

## Consequences
* **Positive**: Classification explanations become **instantaneous (sub-second)** because they are computed in a single forward pass alongside prediction.
* **Positive**: Zero extra dependencies or complex runtime requirements.
* **Negative**: Attention weights represent *association* rather than strict *causality* of prediction, but they are highly effective for highlighting relevant risk-associated keywords (e.g., "bank", "wire", "agreement").
