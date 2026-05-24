import torch
import numpy as np

class AttentionExplainer:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def explain(self, text: str) -> dict:
        if not text or not text.strip():
            return {}
            
        # Resolve active model and tokenizer
        if self.classifier.has_custom_checkpoint:
            model = self.classifier.model
            tokenizer = self.classifier.tokenizer
            is_bart = False
        else:
            # Under Zero-shot fallback, resolve model/tokenizer from pipeline
            model = self.classifier.pipeline.model
            tokenizer = self.classifier.pipeline.tokenizer
            is_bart = True
            
        device = model.device
        
        # Tokenize and execute inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # Extract self-attention layers
        if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
            attentions = outputs.encoder_attentions
        elif hasattr(outputs, "attentions") and outputs.attentions is not None:
            attentions = outputs.attentions
        else:
            # Safe fallback using base model
            base_outputs = model.base_model(**inputs, output_attentions=True)
            attentions = base_outputs.attentions
            
        # Get final attention layer: shape (batch_size, num_heads, seq_len, seq_len)
        final_att = attentions[-1][0].cpu()  # shape (num_heads, seq_len, seq_len)
        
        # Average attention weights across all heads
        mean_att = final_att.mean(dim=0)  # shape (seq_len, seq_len)
        
        # Sum of attention received by each token from all other tokens in the sequence
        token_scores = mean_att.sum(dim=0).numpy()  # shape (seq_len,)
        
        # Reconstruct BPE or WordPiece subwords back to standard words
        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        word_scores = {}
        current_word = ""
        current_scores = []
        
        for token, score in zip(tokens, token_scores):
            is_special = token in {
                "<s>", "</s>", "<pad>", "<unk>", "<mask>", 
                "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"
            }
            if is_special:
                continue
                
            if is_bart:
                # BART uses Byte-Pair Encoding (BPE) with special spacing character 'Ġ' (u0120)
                if token.startswith("Ġ"):
                    if current_word:
                        word_scores[current_word.lower()] = float(np.mean(current_scores))
                    current_word = token[1:]
                    current_scores = [score]
                else:
                    current_word += token
                    current_scores.append(score)
            else:
                # DistilBERT uses WordPiece tokenization where continuation tokens start with '##'
                if token.startswith("##"):
                    current_word += token[2:]
                    current_scores.append(score)
                else:
                    if current_word:
                        word_scores[current_word.lower()] = float(np.mean(current_scores))
                    current_word = token
                    current_scores = [score]
                    
        # Flush last token
        if current_word:
            word_scores[current_word.lower()] = float(np.mean(current_scores))
            
        if not word_scores:
            return {}
            
        # Min-Max Normalization (Scale importance scores between 0.0 and 1.0)
        vals = list(word_scores.values())
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v
        
        normalized_scores = {}
        for word, val in word_scores.items():
            # Clean symbols or punctuation from keys
            clean_word = "".join(c for c in word if c.isalnum())
            if not clean_word:
                continue
                
            if range_v > 0:
                normalized_scores[clean_word] = float((val - min_v) / range_v)
            else:
                normalized_scores[clean_word] = 1.0
                
        return normalized_scores
