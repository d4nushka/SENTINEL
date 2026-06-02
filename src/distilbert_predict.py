"""
distilbert_predict.py
=====================
Loads fine-tuned DistilBERT model and runs inference on new reviews.
F1=0.9782 | Kappa=0.9561 | AUC=0.9980
"""
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

HF_MODEL  = "d4nushka/fake-review-detector-distilbert"
LOCAL_PATH = "outputs/distilbert_full"
MAX_LEN    = 128

def load_distilbert():
    """Load from local if available, else download from HuggingFace Hub."""
    import os
    device = torch.device('cpu')
    source = LOCAL_PATH if os.path.exists(LOCAL_PATH) else HF_MODEL
    print(f"[distilbert] Loading from: {source}")
    tokenizer = DistilBertTokenizer.from_pretrained(source)
    model     = DistilBertForSequenceClassification.from_pretrained(source)
    model.eval()
    model = model.to(device)
    print("[distilbert] Model ready!")
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
    """Predict fake probability for a single review."""
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
        probs     = torch.softmax(outputs.logits, dim=1)
        fake_prob = probs[0][1].item()
    return fake_prob, int(fake_prob > 0.5)

def predict_batch(texts, model, tokenizer, device, batch_size=16):
    """Predict for multiple reviews at once."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    return all_probs

if __name__ == "__main__":
    model, tokenizer, device = load_distilbert()

    test_reviews = [
        "This product is absolutely AMAZING!!! Best purchase EVER! Changed my life!!!",
        "Decent product for the price. Shipping was slow but it does what it says.",
        "This pillow saved my back. I love the look and feel of this pillow.",
        "Terrible experience. Product stopped working after 2 days. Very disappointed.",
        "WOW just WOW!! Best thing I have ever purchased in my entire life. PERFECT!!!",
    ]

    print("\n── DistilBERT Predictions ───────────────────────")
    for review in test_reviews:
        prob, pred = predict(review, model, tokenizer, device)
        label = "FAKE" if pred == 1 else "REAL"
        print(f"  [{label} {prob:.1%}] {review[:70]}")