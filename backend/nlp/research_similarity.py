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
            'full_text': paper_content
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
            if match:
                content = re.sub(r'^#+\s*\w+\s*', '', match.group(0), flags=re.IGNORECASE)
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
                self.word_lemmatizer.lemmatize(token) if use_lemmatization else token
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
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score

    def calculate_transformer_similarity(self, text1, text2):
        text1_chunks = self._chunk_text(text1)
        text2_chunks = self._chunk_text(text2)
        
        similarities = []
        for chunk1 in text1_chunks:
            for chunk2 in text2_chunks:
                try:
                    similarity = self._calculate_bert_similarity(chunk1, chunk2)
                    similarities.append(similarity)
                except Exception as e:
                    print(f"Error in chunk similarity calculation: {e}")
                    similarities.append(0.0)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity

    def calculate_bert_similarity(self, text1, text2):
        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity_score
        except Exception as e:
            print(f"Error in BERT similarity calculation for chunk: {e}")
            return 0.0

def research_similarity():
    paper1_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity-2.md'
    paper2_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity.md'
    
    get_papers = GetPapers()
    paper1, paper2 = get_papers.load_papers(paper1_path, paper2_path)
    
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
            tfidf_score = similarity_calculator.calculate_tfidf_similarity(text1, text2)
            bert_score = similarity_calculator.calculate_bert_similarity(text1, text2)
            
            print(f"{section.capitalize():<15}")
            print(f"Individual Scores: \nTF-IDF: {tfidf_score:.4f}, "
                  f"BERT: {bert_score:.4f}")
        else:
            print(f"{section.capitalize():<15} : No content available")
        print()

