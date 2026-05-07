import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()


def predict_text_risk(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    # sentiment model output
    negative = probabilities[0][0].item()
    neutral = probabilities[0][1].item()
    positive = probabilities[0][2].item()

    # disasters correlate strongly with negative sentiment
    disaster_score = negative

    # boost disaster keywords
    disaster_keywords = [
        "flood", "flooding",
        "heavy rainfall",
        "cyclone",
        "storm",
        "overflow",
        "waterlogging",
        "dam break",
        "landslide"
    ]

    boost = 0
    for word in disaster_keywords:
        if word in text.lower():
            boost += 0.15

    risk_score = min(1.0, disaster_score + boost)

    return float(risk_score)