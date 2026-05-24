import pytest
from models.classifier import ComplianceClassifier

def test_financial_crime_zero_shot_behavior():
    """
    Behavior 1: A text detailing financial crimes (e.g. money laundering, securities fraud)
    must be classified as 'Financial Crime' with high confidence.
    """
    classifier = ComplianceClassifier()
    
    # Financial crime domain text
    text = (
        "The regulatory audit uncovered multiple illegal offshore transactions, structured "
        "deposits designed to evade banking disclosures, and a massive money laundering "
        "scheme utilizing shell corporations to funnel illicit capital."
    )
    
    result = classifier.classify(text)
    
    assert "category" in result
    assert "confidence" in result
    assert result["category"] == "Financial Crime"
    assert result["confidence"] > 0.4

def test_fraud_zero_shot_behavior():
    """
    Behavior 2: A text detailing corporate bribery, fake invoicing, or phishing scams
    must be classified as 'Fraud' with high confidence.
    """
    classifier = ComplianceClassifier()
    text = (
        "The vendor submitted falsified billing invoices for services that were never performed "
        "and bribed the procurement manager with off-the-books payments to approve the expense."
    )
    result = classifier.classify(text)
    assert result["category"] == "Fraud"
    assert result["confidence"] > 0.4

def test_regulatory_breach_zero_shot_behavior():
    """
    Behavior 3: A text describing missing SEC filings or FTC standards violations
    must be classified as 'Regulatory Breach' with high confidence.
    """
    classifier = ComplianceClassifier()
    text = (
        "The corporation failed to submit its required annual environmental impact statements "
        "and exceeded the statutory limits on carbon discharge mandated by state agency rules."
    )
    result = classifier.classify(text)
    assert result["category"] == "Regulatory Breach"
    assert result["confidence"] > 0.4

def test_cli_nonexistent_file_resilience():
    """
    Behavior 4: Running the CLI classify mode with a nonexistent file path
    must exit gracefully with a clear error message.
    """
    import subprocess
    import sys
    
    # Run CLI using subprocess
    python_bin = sys.executable
    cmd = [python_bin, "main.py", "--mode", "classify", "--input", "docs/nonexistent.txt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode != 0
    assert "Error: File 'docs/nonexistent.txt' does not exist." in result.stdout or "Error: File 'docs/nonexistent.txt' does not exist." in result.stderr

def test_cli_successful_classification(tmp_path):
    """
    Behavior 4: Running the CLI with a valid file must output structured results cleanly.
    """
    import subprocess
    import sys
    
    # Write a test file
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text(
        "The vendor submitted falsified billing invoices for services that were never performed "
        "and bribed the procurement manager with off-the-books payments."
    )
    
    python_bin = sys.executable
    cmd = [python_bin, "main.py", "--mode", "classify", "--input", str(test_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Category:   Fraud" in result.stdout
    assert "Confidence:" in result.stdout
    assert "--- Attention Keyword Importance ---" in result.stdout
    assert "bribed" in result.stdout or "falsified" in result.stdout
    assert "No entities extracted (Date, Org, Money)." in result.stdout

def test_cli_successful_classification_with_entities(tmp_path):
    """
    Behavior 3: Running the CLI with a valid file containing entities must list them cleanly in a table format.
    """
    import subprocess
    import sys
    
    # Write a test file containing entities
    test_file = tmp_path / "test_doc_ent.txt"
    test_file.write_text(
        "On June 15, 2026, ACME Corp. lost $5,000,000 in structured banking capital."
    )
    
    python_bin = sys.executable
    cmd = [python_bin, "main.py", "--mode", "classify", "--input", str(test_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Category:   Financial Crime" in result.stdout
    assert "--- Attention Keyword Importance ---" in result.stdout
    assert "capital" in result.stdout or "structured" in result.stdout or "banking" in result.stdout
    assert "--- Extracted Named Entities ---" in result.stdout
    assert "ACME Corp." in result.stdout
    assert "June 15, 2026" in result.stdout
    assert "5,000,000" in result.stdout


def test_spacy_entity_extraction_behavior():
    """
    Behavior 1: CompliancePreprocessor must identify and extract ORG, DATE, and MONEY entities
    from text with correct character offset bounds.
    """
    from preprocessing.nlp_pipeline import CompliancePreprocessor
    preprocessor = CompliancePreprocessor()
    
    text = "On June 15, 2026, ACME Corp. lost $5,000,000 in structured banking capital."
    entities = preprocessor.extract_entities(text)
    
    # Extract just labels and texts for verification
    extracted = [(ent["text"], ent["label"]) for ent in entities]
    
    # We expect:
    # "June 15, 2026" as DATE
    # "ACME Corp." as ORG
    # "$5,000,000" as MONEY
    assert ("June 15, 2026", "DATE") in extracted
    assert ("ACME Corp.", "ORG") in extracted
    assert ("5,000,000", "MONEY") in extracted
    
    # Verify offsets correspond to the text
    for ent in entities:
        assert text[ent["start"]:ent["end"]] == ent["text"]

def test_spacy_empty_and_no_entities_behavior():
    """
    Behavior 2: CompliancePreprocessor should return an empty list [] gracefully
    when passed empty text, whitespace-only text, or text without target entities.
    """
    from preprocessing.nlp_pipeline import CompliancePreprocessor
    preprocessor = CompliancePreprocessor()
    
    assert preprocessor.extract_entities("") == []
    assert preprocessor.extract_entities("   ") == []
    
    # Text with no DATE, ORG, or MONEY
    text = "Hello world! This is a simple sentence."
    assert preprocessor.extract_entities(text) == []

def test_attention_explainability_behavior():
    """
    Behavior 1: AttentionExplainer must extract token self-attentions during classification,
    filter special tokens, and return a dictionary of normalized word importances.
    """
    from explainability.lime_analysis import AttentionExplainer
    classifier = ComplianceClassifier()
    explainer = AttentionExplainer(classifier)
    
    text = "The vendor submitted falsified billing invoices and bribed the procurement manager."
    explanation = explainer.explain(text)
    
    # Assert output shape
    assert isinstance(explanation, dict)
    assert len(explanation) > 0
    
    # Assert scores are normalized between 0.0 and 1.0
    for token, score in explanation.items():
        assert 0.0 <= score <= 1.0
        
    # Check that domain keywords (e.g. 'bribed', 'falsified') are present and have non-zero importance
    keywords = {"bribed", "falsified", "billing", "invoices"}
    found_keywords = [kw for kw in keywords if kw in explanation]
    assert len(found_keywords) > 0

def test_attention_explainability_empty_behavior():
    """
    Behavior 2: AttentionExplainer must return an empty dictionary gracefully
    when passed empty strings, whitespace-only text, or non-alphabetic/punctuation-only text.
    """
    from explainability.lime_analysis import AttentionExplainer
    classifier = ComplianceClassifier()
    explainer = AttentionExplainer(classifier)
    
    assert explainer.explain("") == {}
    assert explainer.explain("   ") == {}
    assert explainer.explain("!!!") == {}

def test_rag_ingestion_and_persistence_behavior(tmp_path):
    """
    Behavior 1: ComplianceRAG must chunk text, generate sentence embeddings,
    and build a persistent FAISS index folder on disk.
    """
    import os
    from models.rag_pipeline import ComplianceRAG
    
    # Use temporary directory for testing persistent FAISS
    index_dir = str(tmp_path / "faiss_index")
    rag = ComplianceRAG(index_dir=index_dir)
    
    text = (
        "Under Section 402 of the Federal Clean Water Standard, the statutory carbon "
        "discharge limits are set at a maximum of 50 parts per million. Any corporate "
        "facility violating this rule is subject to administrative warnings and immediate "
        "remediation fines up to $250,000 per violation day."
    )
    
    chunks_added = rag.ingest_document(text, doc_name="EnvCompliance")
    
    assert chunks_added > 0
    # Check that FAISS index files are saved on disk
    assert os.path.exists(os.path.join(index_dir, "index.faiss"))
    assert os.path.exists(os.path.join(index_dir, "index.pkl"))

def test_rag_semantic_retrieval(tmp_path):
    """
    Behavior 2: ComplianceRAG must perform semantic similarity searches
    and return correct retrieved chunks matching a natural query.
    """
    from models.rag_pipeline import ComplianceRAG
    index_dir = str(tmp_path / "faiss_index")
    rag = ComplianceRAG(index_dir=index_dir)
    
    text = "Statutory remediation fines are evaluated at $250,000 per violation day for corporate discharge."
    rag.ingest_document(text, doc_name="FineMandate")
    
    # Query semantically (discharge limits matches discharge, fines matches fines)
    results = rag.db.similarity_search("statutory remediation fines", k=1)
    
    assert len(results) == 1
    assert "remediation fines" in results[0].page_content
    assert results[0].metadata["source"] == "FineMandate"

def test_rag_offline_fallback_generation(tmp_path):
    """
    Behavior 3: ComplianceRAG must automatically trigger the local fallback generator
    when no Groq API Key is present, returning a formatted markdown report.
    """
    from models.rag_pipeline import ComplianceRAG
    index_dir = str(tmp_path / "faiss_index")
    rag = ComplianceRAG(index_dir=index_dir)
    
    text = "Statutory remediation fines are evaluated at $250,000 per violation day for corporate discharge."
    rag.ingest_document(text, doc_name="FineMandate")
    
    # Query with empty API key to force local synthesizer report
    res = rag.query("What is the cost of corporate discharge violations?", groq_api_key="")
    
    assert res["mode"] == "fallback"
    assert len(res["retrieved_chunks"]) > 0
    assert "# RAG Compliance Summary Report" in res["answer"]
    assert "### Question Asked" in res["answer"]
    assert "FineMandate" in res["answer"]
    assert "remediation fines" in res["answer"]

def test_rag_online_generation_mocked(tmp_path):
    """
    Behavior 4: ComplianceRAG should call the Groq completions API and return conversational
    LLM answers under online mode when a valid API key is present.
    """
    from unittest.mock import patch, MagicMock
    from models.rag_pipeline import ComplianceRAG
    
    index_dir = str(tmp_path / "faiss_index")
    rag = ComplianceRAG(index_dir=index_dir)
    
    text = "Statutory remediation fines are evaluated at $250,000 per violation day for corporate discharge."
    rag.ingest_document(text, doc_name="FineMandate")
    
    # Mock Groq client and create completion response structure
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "According to Section 402, corporate facilities face up to $250,000 in remediation fines."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    
    with patch("groq.Groq", return_value=mock_client):
        # Query with API key, causing it to trigger the online path
        res = rag.query("What is the cost of corporate discharge violations?", groq_api_key="mock_key_123")
        
        assert res["mode"] == "online"
        assert len(res["retrieved_chunks"]) > 0
        assert res["answer"] == "According to Section 402, corporate facilities face up to $250,000 in remediation fines."
        # Verify Groq chat completions create was indeed called
        mock_client.chat.completions.create.assert_called_once()

def test_cli_query_resilience():
    """
    Behavior 5: Running the CLI query mode without a --question argument must fail with a clear error.
    """
    import subprocess
    import sys
    
    python_bin = sys.executable
    cmd = [python_bin, "main.py", "--mode", "query"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode != 0
    assert "Error: --question <question_text> is required in query mode." in result.stdout or "Error: --question <question_text> is required in query mode." in result.stderr

def test_cli_query_successful_run(tmp_path):
    """
    Behavior 5: Running the CLI query mode with a valid question must output RAG responses cleanly.
    """
    import os
    import subprocess
    import sys
    from models.rag_pipeline import ComplianceRAG
    
    # Ingest text to default index so CLI can load it (clean default faiss_index)
    # We will override the default index path temporarily or just let it write to data/faiss_index
    # To prevent side effects on dev environment, let's patch the CLI or run it on the default index
    default_index = "data/faiss_index"
    rag = ComplianceRAG(index_dir=default_index)
    
    text = "The SEC requires corporate disclosures for executive stock sales within 48 hours."
    rag.ingest_document(text, doc_name="SECDeadline")
    
    python_bin = sys.executable
    cmd = [python_bin, "main.py", "--mode", "query", "--question", "What is the SEC executive sales disclosure deadline?"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "--- RAG Query Response ---" in result.stdout
    assert "SECDeadline" in result.stdout
    assert "48 hours" in result.stdout
    
    # Clean up index
    try:
        import shutil
        shutil.rmtree(default_index)
    except Exception:
        pass

def test_dataset_mapping_behavior():
    """
    Behavior 1: load_and_map_dataset must download the SCOTUS dataset,
    filter relevant issue categories, and deterministic map to labels {0, 1, 2}.
    """
    from data.load_dataset import load_and_map_dataset
    
    # We will test dataset mapping on a tiny slice to keep tests blazingly fast
    dataset = load_and_map_dataset(tiny_slice=True)
    
    assert "train" in dataset
    assert "validation" in dataset
    
    # Check that labels are mapped and contain only 0, 1, or 2
    train_labels = dataset["train"]["label"]
    unique_labels = set(train_labels)
    assert unique_labels.issubset({0, 1, 2})

def test_classifier_training_and_checkpoint_loading(tmp_path):
    """
    Behavior 2: train_classifier must successfully compile and execute a fine-tuning epoch on a tiny dataset
    and serialize model weights on disk.
    Behavior 3: ComplianceClassifier must auto-load this custom checkpoint and return predictions.
    """
    import os
    from datasets import Dataset, DatasetDict
    from models.classifier import train_classifier, ComplianceClassifier
    
    # 1. Create a tiny mock dataset
    mock_data = {
        "text": [
            "We indictment money laundering offshore evade disclose.",
            "bribery invoice procurement manager falsified off-the-books payments.",
            "environmental statutory carbon discharge limit breach regulatory agency EPA rules."
        ],
        "label": [0, 1, 2]
    }
    ds = Dataset.from_dict(mock_data)
    tiny_dataset = DatasetDict({"train": ds, "validation": ds})
    
    checkpoint_dir = str(tmp_path / "best_model")
    
    # 2. Run fast training compile test (1 epoch, batch size 1)
    train_classifier(output_dir=checkpoint_dir, epochs=1, batch_size=1, tiny_dataset=tiny_dataset)
    
    # Assert checkpoint files are saved on disk
    assert os.path.exists(os.path.join(checkpoint_dir, "config.json"))
    assert os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin"))
    
    # 3. Assert Behavior 3: ComplianceClassifier seamlessly auto-loads custom checkpoint
    classifier = ComplianceClassifier(checkpoint_dir=checkpoint_dir)
    assert classifier.has_custom_checkpoint is True
    assert classifier.pipeline is None  # Zero-shot pipeline is disabled
    
    # Test inference using the custom loaded checkpoint
    res = classifier.classify("money laundering offshore evade")
    assert "category" in res
    assert "confidence" in res
    assert "attention" in res

def test_cli_train_execution():
    """
    Behavior 4: Running the CLI train mode must invoke the trainer and exit gracefully.
    """
    import pytest
    from unittest.mock import patch
    from main import main
    
    with patch("sys.argv", ["main.py", "--mode", "train"]):
        with patch("models.classifier.train_classifier") as mock_train:
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_train.assert_called_once()


def test_streamlit_app_syntax():
    """
    Behavior 5: Test that the Streamlit dashboard app syntactically compiles successfully using AST parser.
    """
    import ast
    import os
    
    app_path = os.path.join(os.path.dirname(__file__), "../dashboard/streamlit_app.py")
    assert os.path.exists(app_path), "streamlit_app.py does not exist at expected path."
    
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    try:
        ast.parse(content)
    except SyntaxError as e:
        pytest.fail(f"streamlit_app.py failed syntax parsing check: {e}")

