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




