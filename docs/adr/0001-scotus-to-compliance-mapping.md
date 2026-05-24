# ADR 0001: Mapping LexGLUE SCOTUS Issue Areas to Compliance Risk Categories

## Status
Proposed (Pending approval in current plan)

## Context
We need a robust, labeled text dataset of at least 10,000 documents to fine-tune a DistilBERT classifier for our three target compliance risk categories:
1. **Financial Crime**
2. **Fraud**
3. **Regulatory Breach**

The LexGLUE benchmark is a standard legal/regulatory text corpus, but it does not contain native labels matching these three exact categories. We evaluated the two main candidate subsets:
* **ECtHR**: European Court of Human Rights cases. The categories correspond to human rights articles (e.g., Article 3: torture, Article 6: fair trial), which have little relevance to corporate financial compliance.
* **SCOTUS**: Supreme Court of the United States cases. Labeled by general legal issue areas (e.g., Economic Activity, Criminal Procedure, Federalism, Judicial Power).

## Decision
We will use the **LexGLUE SCOTUS** subset for fine-tuning our DistilBERT classifier. We will define an explicit, deterministic mapping from SCOTUS issue area IDs to our target risk classes:
* **Financial Crime**: Sourced from SCOTUS cases under *Economic Activity* dealing with banking regulations, securities trading, antitrust violations, and tax compliance.
* **Fraud**: Sourced from SCOTUS cases dealing with corruption, bribery, mail/wire fraud, theft, and corporate deception.
* **Regulatory Breach**: Sourced from SCOTUS cases dealing with administrative agency jurisdictions (SEC, FTC, IRS rules), interstate commerce, and statutory compliance disputes.

Documents that do not fit into these categories will be filtered out to ensure high label quality.

## Alternatives Considered
* **ECtHR Subset**: Rejected because the legal domains (human rights and civil liberties) do not translate well to financial compliance risks.
* **Zero-shot LLM Labeling**: Rejected due to high token cost and potential inconsistencies compared to a deterministic ground-truth mapping from a benchmark dataset.

## Consequences
* **Positive**: We get high-quality, professional legal texts with labels that map cleanly to real-world corporate compliance categories.
* **Negative**: The training set size might shrink slightly after filtering out irrelevant SCOTUS categories (e.g., First Amendment, civil rights), but SCOTUS is large enough that we can still exceed our 10K document target.
* **Risk**: Any changes to this mapping in the future will require a full model re-training run.
