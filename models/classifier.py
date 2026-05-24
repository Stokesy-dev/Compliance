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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir, 
                attn_implementation="eager"
            )
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
                
            # Load BART large MNLI zero-shot pipeline with eager attention
            self.pipeline = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli", 
                device=device,
                model_kwargs={"attn_implementation": "eager"}
            )
            
    def classify(self, text: str) -> dict:
        # Dynamically import to avoid circular dependency
        from explainability.lime_analysis import AttentionExplainer
        explainer = AttentionExplainer(self)
        attention_scores = explainer.explain(text)
        
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
                "attention": attention_scores
            }
        else:
            # Zero-shot classification
            res = self.pipeline(text, candidate_labels=self.labels)
            best_label = res["labels"][0]
            best_score = res["scores"][0]
            return {
                "category": best_label,
                "confidence": float(best_score),
                "attention": attention_scores
            }

def train_classifier(output_dir: str = "models/checkpoints/best_model", epochs: int = 3, batch_size: int = 8, tiny_dataset=None):
    """
    Fine-tunes distilbert-base-uncased on corporate compliance risk classes.
    Saves the final trained model weights and configurations to output_dir.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
    from data.load_dataset import load_and_map_dataset
    
    # 1. Load labeled dataset
    if tiny_dataset is not None:
        dataset = tiny_dataset
    else:
        print("Downloading SCOTUS dataset and applying compliance mappings...")
        dataset = load_and_map_dataset()
        
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Tokenize inputs
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 3. Load DistilBERT model with eager attention (required for extraction highlights)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        attn_implementation="eager"
    )
    
    # 4. Set up Training parameters (use CPU for safe test executions)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch" if "validation" in tokenized_dataset else "no",
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        disable_tqdm=True,
        use_cpu=True
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        processing_class=tokenizer
    )
    
    # 6. Fine-tune and save checkpoints
    print("Beginning model fine-tuning...")
    trainer.train()
    print(f"Fine-tuning complete. Saving weights to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

