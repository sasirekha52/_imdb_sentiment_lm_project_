from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_dir: str | Path, max_length: int = 256):
        model_dir = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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
