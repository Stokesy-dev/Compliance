import spacy

class CompliancePreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Dynamic fallback to ensure load succeeded
            import sys
            import subprocess
            python_bin = sys.executable
            subprocess.run([python_bin, "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
            
    def extract_entities(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []
            
        doc = self.nlp(text)
        entities = []
        target_labels = {"DATE", "ORG", "MONEY"}
        
        for ent in doc.ents:
            if ent.label_ in target_labels:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities
