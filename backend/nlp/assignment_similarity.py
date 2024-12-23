import re
import nltk
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer

class PlagiarismDetector:
    def __init__(self, ngram_range=(1, 3), use_bert=False):
        self.ngram_range = ngram_range
        self.use_bert = use_bert  # Whether to use BERT or TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            analyzer='word',
            lowercase=True,
            strip_accents='unicode'
        )
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Load pre-trained Sentence-BERT model for semantic similarity
        if self.use_bert:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess(self, text):
        """
        Preprocess text by removing references, tokenizing, lemmatizing, and removing stopwords.
        Optimized to reduce redundant operations and process tokens in batches.
        """
        # Remove references section but not the introduction
        text = re.sub(r'(References|Table of Contents).*', '', text, flags=re.DOTALL)

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # POS tag all tokens in one call (batch process for speed)
        tagged_tokens = nltk.pos_tag(tokens)

        # Lemmatize and remove stopwords
        processed_tokens = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for word, tag in tagged_tokens
            if word.lower() not in self.stop_words and word.isalpha()
        ]

        # Join tokens back into string
        return ' '.join(processed_tokens)

    def get_wordnet_pos(self, nltk_tag):
        """
        Map NLTK POS tags to WordNet format for lemmatization.
        """
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(nltk_tag[0].upper(), wordnet.NOUN)

    def extract_text_from_pdf(self, file_path):
        """
        Extract text from PDF file using PyPDF2. Optimized to process large PDFs page-by-page.
        """
        text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return ' '.join(text)

    def calculate_similarity(self, pdf1, pdf2):
        """
        Calculate similarity between two PDF documents using either TF-IDF or BERT.
        """
        # Preprocess both PDFs
        doc1 = self.preprocess(self.extract_text_from_pdf(pdf1))
        doc2 = self.preprocess(self.extract_text_from_pdf(pdf2))

        if self.use_bert:
            # Use Sentence-BERT to compute semantic similarity
            embeddings = self.bert_model.encode([doc1, doc2])
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        else:
            # Use TF-IDF and cosine similarity, fitting the vectorizer only once
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([doc1, doc2])
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return similarity_score

if __name__ == "__main__":
    detector = PlagiarismDetector(ngram_range=(1, 1), use_bert=False)

    # Replace these with your PDF paths
    pdf1 = "documents/test1.pdf"
    pdf2 = "documents/classification-of-human-and-ai-generated-texts-investigating.pdf"

    similarity = detector.calculate_similarity(pdf1, pdf2)
    print(f"Similarity score: {similarity:.4f}")
