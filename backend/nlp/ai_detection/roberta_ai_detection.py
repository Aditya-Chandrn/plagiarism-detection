from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Load the model and tokenizer
model_name = "roberta-large-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def roberta_ai_detection(file_path):
    from routers.utils import read_md_file
    text = read_md_file(file_path)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)

    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze()

    # Print results
    labels = ["Human-written", "AI-generated"]
    prediction = labels[torch.argmax(probabilities).item()]
    confidence = probabilities.max().item()

    print("--------- x  ROBERTA AI DETECTION  x ---------")
    print(f"Prediction: {prediction} | Confidence: {confidence}")

    return confidence
