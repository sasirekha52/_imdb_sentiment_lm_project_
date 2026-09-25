from pathlib import Path
import torch
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_dir: str | Path, max_length: int = 256):
        # Directly download and load from the official Hugging Face hub
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
        # Manually map the internal configuration labels for your application metrics
        self.model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probabilities).item())
        label = self.model.config.id2label.get(idx, str(idx))
        return {
            "label": label,
            "confidence": float(probabilities[idx].item()),
            "probabilities": {
                self.model.config.id2label.get(i, str(i)): float(probabilities[i].item())
                for i in range(len(probabilities))
            },
        }
