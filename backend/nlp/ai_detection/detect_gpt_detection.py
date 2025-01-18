import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load the model and tokenizer
model_name = "gpt2"  # Use a model capable of providing log probabilities
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Helper function to calculate log probability of a text
def compute_log_probability(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_prob = -outputs.loss.item()  # Negative loss as the log probability
    return log_prob

# Step 3: Generate perturbed versions of the text
def generate_perturbations(text, num_perturbations=5):
    words = text.split()
    synonym_dict = {
        "good": ["great", "excellent", "nice"],
        "bad": ["terrible", "awful", "poor"],
        "quick": ["fast", "rapid", "speedy"],
        "slow": ["sluggish", "lethargic", "unhurried"]
    }

    perturbed_texts = []
    for _ in range(num_perturbations):
        perturbed_words = [
            random.choice(synonym_dict.get(word, [word])) if random.random() < 0.3 else word
            for word in words
        ]
        perturbed_texts.append(" ".join(perturbed_words))
    return perturbed_texts

# Main function to calculate curvature and classify text
def detect_ai_generated_text(text, num_perturbations=5, threshold=0.5):
    # Step 2: Calculate log probability of the original text
    original_log_prob = compute_log_probability(text)

    # Step 4: Generate perturbed texts and compute their log probabilities
    perturbed_texts = generate_perturbations(
        text, num_perturbations=num_perturbations)
    perturbed_log_probs = [compute_log_probability(
        perturbed) for perturbed in perturbed_texts]

    # Step 5: Check for empty perturbation list
    if not perturbed_log_probs:
        raise ValueError("No perturbed log probabilities available.")

    # Step 6: Calculate average log probability of perturbed texts
    avg_perturbed_log_prob = sum(
        perturbed_log_probs) / len(perturbed_log_probs)

    # Step 7: Calculate curvature score
    curvature_score = original_log_prob - avg_perturbed_log_prob

    # Step 8: Classify based on curvature score
    if curvature_score > threshold:
        return "AI-generated", curvature_score
    else:
        return "Human-written", curvature_score

# Test the function with example text
def detect_gpt_main(file_path):
    from routers.utils import read_md_file
    try:
        text = read_md_file(file_path)
        if not text:
            raise ValueError("The input text file is empty or cannot be read.")
        classification, curvature_score = detect_ai_generated_text(text)

        print("--------- x  DETECT GPT  x ---------")
        print(f"Classification: {classification}")
        print(f"Curvature Score: {curvature_score}")
        return curvature_score
    except Exception as e:
        print(f"Error: {e}")
        return None
