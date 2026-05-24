import os
import torch
from transformers import pipeline

class ComplianceClassifier:
    def __init__(self, checkpoint_dir: str = "models/checkpoints/best_model"):
        self.checkpoint_dir = checkpoint_dir
        self.has_custom_checkpoint = os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir)
        
        # Labels mapping
        self.labels = ["Financial Crime", "Fraud", "Regulatory Breach"]
        
        if self.has_custom_checkpoint:
            # Load fine-tuned DistilBERT checkpoint (Slice 5 implementation)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            self.pipeline = None
        else:
            # Load Zero-Shot Fallback
            # Select best hardware accelerator
            if torch.cuda.is_available():
                device = 0
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            # Load BART large MNLI zero-shot pipeline
            self.pipeline = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli", 
                device=device
            )
            
    def classify(self, text: str) -> dict:
        if self.has_custom_checkpoint:
            # Custom model inference
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
            
            max_idx = probs.index(max(probs))
            return {
                "category": self.labels[max_idx],
                "confidence": probs[max_idx],
                "attention": {}
            }
        else:
            # Zero-shot classification
            res = self.pipeline(text, candidate_labels=self.labels)
            best_label = res["labels"][0]
            best_score = res["scores"][0]
            return {
                "category": best_label,
                "confidence": float(best_score),
                "attention": {}
            }
