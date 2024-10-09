import nltk

# Function to download required NLTK packages
def download_nltk_data():
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for dataset in datasets:
        print(f"Downloading {dataset} if not already present...")
        nltk.download(dataset)

# Call the download function
download_nltk_data()

print("NLTK data setup completed!")