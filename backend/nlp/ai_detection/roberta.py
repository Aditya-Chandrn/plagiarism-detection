from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "roberta-large-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_ai_generated_content(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze()

    # Print results
    labels = ["Human-written", "AI-generated"]
    prediction = labels[torch.argmax(probabilities).item()]
    confidence = probabilities.max().item()

    return prediction, confidence
