import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


class GetPapers:
    def load_papers(self, paper1_path, paper2_path):

        try:
            with open(paper1_path, "r", encoding='utf-8') as f1:
                paper1_content = f1.read()
            with open(paper2_path, "r", encoding='utf-8') as f2:
                paper2_content = f2.read()

            return paper1_content, paper2_content

        except Exception as e:
            print(f'Error reading files: {e}')
            return None, None

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.word_lemmatizer = WordNetLemmatizer()

    def extract_sections(self, paper_content):
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'full_text': paper_content,
        }

        section_patterns = {
            'abstract': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:abstract).*?(?=\n#+|$)',
            'introduction': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:introduction|background).*?(?=\n#+|$)',
            'methodology': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:methodology|methods|materials and methods).*?(?=\n#+|$)',
            'results': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:results|findings).*?(?=\n#+|$)',
            'discussion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:discussion).*?(?=\n#+|$)',
            'conclusion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:conclusion|final thoughts).*?(?=\n#+|$)'
        }

        for section, pattern in section_patterns.items():
            match = re.search(pattern, paper_content, re.DOTALL)
            match = re.search(pattern, paper_content, re.DOTALL)
            if match:
                content = re.sub(r'^#+\s*\w+\s*', '',
                                 match.group(0), flags=re.IGNORECASE)
                sections[section] = content.strip()

        return sections

    def preprocess_text(self, text, use_lemmatization=True):
        if not text:
            return ''

        try:
            text = re.sub(r'[^a-zA-Z0-9\s.,]', ' ', text)
            text = text.lower()

            tokens = nltk.word_tokenize(text)
            filtered_tokens = [
                self.word_lemmatizer.lemmatize(
                    token) if use_lemmatization else token
                for token in tokens if token not in self.stop_words
            ]

            return ' '.join(filtered_tokens)

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return ''

class SimilarityCalculator:
    def __init__(self, bert_model='all-MiniLM-L6-v2', max_chunk_length=510):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bert_model = SentenceTransformer(bert_model)
        self.max_chunk_length = max_chunk_length

    def _chunk_text(self, text):
        """Splits text into approximate word chunks before tokenizing."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.max_chunk_length):
            chunk = ' '.join(words[i:i + self.max_chunk_length])
            chunks.append(chunk)
        return chunks

    def calculate_tfidf_similarity(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score

    def calculate_transformer_similarity(self, text1, text2):
        text1_chunks = self._chunk_text(text1)
        text2_chunks = self._chunk_text(text2)

        similarities = []
        for chunk1 in text1_chunks:
            for chunk2 in text2_chunks:
                try:
                    similarity = self.calculate_bert_similarity(chunk1, chunk2)
                    similarities.append(similarity)
                except Exception as e:
                    print(f"Error in chunk similarity calculation: {e}")
                    similarities.append(0.0)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity

    def calculate_bert_similarity(self, text1, text2):
        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity_score = cosine_similarity(
                [embeddings[0]], [embeddings[1]])[0][0]
            return similarity_score
        except Exception as e:
            print(f"Error in BERT similarity calculation for chunk: {e}")
            return 0.0

    def combined_similarity(self, text1, text2, tfidf_weight=0.3, bert_weight=0.7):
        tfidf_score = self.calculate_tfidf_similarity(text1, text2)
        bert_score = self.calculate_transformer_similarity(text1, text2)
        combined_score = (tfidf_weight * tfidf_score) + \
            (bert_weight * bert_score)
        return combined_score, {"TF-IDF": tfidf_score, "BERT": bert_score}
    
    def extract_plagiarized_sentences(self, text1, text2, similarity_threshold=0.6):
        # Ensure NLTK sentence tokenizer is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Tokenize sentences
        sentences1 = nltk.sent_tokenize(text1)
        sentences2 = nltk.sent_tokenize(text2)
        
        # Store plagiarized sentences
        plagiarized_sentences = {
            'sentences': [],
            'details': []
        }
        
        # Check each sentence in text2 against sentences in text1
        for idx2, sentence2 in enumerate(sentences2):
            # Preprocess the sentences
            preprocessor = Preprocessor()
            processed_sentence2 = preprocessor.preprocess_text(sentence2)
            
            # Track best match
            best_match = {
                'similarity': 0,
                'source_sentence': None
            }
            
            # Compare with each sentence in text1
            for idx1, sentence1 in enumerate(sentences1):
                processed_sentence1 = preprocessor.preprocess_text(sentence1)
                
                # Calculate combined similarity
                combined_score, _ = self.combined_similarity(
                    processed_sentence1, 
                    processed_sentence2
                )
                
                # Update best match if current similarity is higher
                if combined_score > best_match['similarity']:
                    best_match['similarity'] = combined_score
                    best_match['source_sentence'] = sentence1
            
            # If similarity exceeds threshold, consider it plagiarized
            if best_match['similarity'] >= similarity_threshold:
                plagiarized_sentences['sentences'].append(sentence2)
                plagiarized_sentences['details'].append({
                    'plagiarized_sentence': sentence2,
                    'source_sentence': best_match['source_sentence'],
                    'similarity_score': best_match['similarity']
                })
        
        return plagiarized_sentences

def research_similarity(path1):
    paper2_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity-2.md'
    # paper2_path = 'C:/College/College Work/plagiarism-detection/backend/documents/research-paper-2.md'
    # paper2_path = 'C:/College/College Work/plagiarism-detection/backend/documents/test-2.md'

    get_papers = GetPapers()
    paper1, paper2 = get_papers.load_papers(path1, paper2_path)

    if not paper1 or not paper2:
        print("Error loading papers")
        return

    preprocessor = Preprocessor()
    sections_paper1 = preprocessor.extract_sections(paper1)
    sections_paper2 = preprocessor.extract_sections(paper2)

    if not sections_paper1 or not sections_paper2:
        print("Error extracting sections")
        return

    similarity_calculator = SimilarityCalculator()

    print('Similarity Score by Sections')
    print('-'*30)

    for section in sections_paper1.keys():
        text1 = preprocessor.preprocess_text(sections_paper1[section])
        text2 = preprocessor.preprocess_text(sections_paper2[section])

        if text1 and text2:
            combined_score, individual_scores = similarity_calculator.combined_similarity(
                text1, text2)
            tfidf_score = similarity_calculator.calculate_tfidf_similarity(
                text1, text2)
            bert_score = similarity_calculator.calculate_bert_similarity(
                text1, text2)

            # print(f"{section.capitalize():<15}")
            # print(f"Individual Scores: \nTF-IDF: {tfidf_score:.4f}, "
            #       f"BERT: {bert_score:.4f}")
            print(
                f"{section.capitalize():<15} : Combined Score: {combined_score:.4f}")
            print(f"Individual Scores: TF-IDF: {individual_scores['TF-IDF']:.4f}, "
                  f"BERT: {individual_scores['BERT']:.4f}")
        else:
            print(f"{section.capitalize():<15} : No content available")
        print()
    
    plagiarized_sentences = similarity_calculator.extract_plagiarized_sentences(
        preprocessor.preprocess_text(paper1), 
        preprocessor.preprocess_text(paper2)
    )
    
    print("\nPlagiarized Sentences:")
    for detail in plagiarized_sentences['details']:
        print(f"Plagiarized Sentence: {detail['plagiarized_sentence']}")
        print(f"Source Sentence: {detail['source_sentence']}")
        print(f"Similarity Score: {detail['similarity_score']:.4f}\n")


    return {"data": {"name": "src1", "url": "http://abc.com"}, "bert_score": bert_score, "tfidf_score": tfidf_score, "score": combined_score}
