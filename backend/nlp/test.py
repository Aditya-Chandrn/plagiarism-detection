import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LongformerModel, LongformerTokenizer
import torch
from typing import Dict, Tuple, List

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
            'abstract': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:abstract|overview|summary).*?(?=\n#+|$)',
            'introduction': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:introduction|background|overview).*?(?=\n#+|$)',
            'methodology': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:methodology|methods|implementation|experiment|materials and methods).*?(?=\n#+|$)',
            'results': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:results|findings|observations).*?(?=\n#+|$)',
            'discussion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:discussion|interpretation|analysis).*?(?=\n#+|$)',
            'conclusion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:conclusion|summary|final thoughts).*?(?=\n#+|$)'
        }

        # Loop over patterns and apply extraction logic for each section
        for section, pattern in section_patterns.items():
            match = re.search(pattern, paper_content, re.DOTALL)
            if match:
                # Remove any heading at the start of the matched content
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

class DocumentChunker:
    def __init__(self, chunk_size: int = 4096, overlap: int = 256):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, tokens: List[int]) -> List[List[int]]:
        chunks = []
        start = 0
        document_length = len(tokens)

        while start < document_length:
            end = min(start + self.chunk_size, document_length)
            chunks.append(tokens[start:end])
            start = end - self.overlap  # Move the start point for the next chunk

            # Break if we’ve reached the end of the document
            if end == document_length:
                break

        return chunks

# class SimilarityCalculator:
#     def __init__(self, model_name: str = 'allenai/longformer-base-4096'):
#         # Initialize Longformer
#         self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
#         self.model = LongformerModel.from_pretrained(model_name)
#         self.model.eval()
        
#         # TF-IDF vectorizer
#         self.tfidf_vectorizer = TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=50000
#         )
        
#         self.chunker = DocumentChunker()
        
#         # GPU support
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#     @torch.no_grad()
#     def calculate_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
#         # Quick TF-IDF check
#         tfidf_score = self._calculate_tfidf_similarity(text1, text2)
        
#         # Early exit for very dissimilar texts
#         if tfidf_score < 0.1:
#             return tfidf_score, {
#                 "tfidf": tfidf_score,
#                 "transformer": 0.0,
#                 "method": "TF-IDF only (low similarity threshold)"
#             }

#         # Calculate transformer similarity
#         transformer_score = self._calculate_transformer_similarity(text1, text2)
        
#         # Weight the scores (adjustable weights)
#         combined_score = (0.3 * tfidf_score) + (0.7 * transformer_score)
        
#         return combined_score, {
#             "tfidf": tfidf_score,
#             "transformer": transformer_score,
#             "method": "Combined TF-IDF and Longformer"
#         }

#     def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
#         try:
#             tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
#             return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#         except Exception as e:
#             print(f"Error in TF-IDF calculation: {str(e)}")
#             return 0.0

#     def _calculate_transformer_similarity(self, text1: str, text2: str) -> float:
#       try:
#         # Get embeddings
#         emb1 = self._get_embedding(text1)
#         emb2 = self._get_embedding(text2)
#         # Calculate similarity
#         similarity = cosine_similarity(emb1.cpu().numpy().reshape(1, -1),emb2.cpu().numpy().reshape(1, -1))[0][0]
#         return float(similarity)
#       except Exception as e:
#         print(f"Error in transformer calculation: {str(e)}")
#         return 0.0
      
#     def _get_embedding(self, text: str) -> torch.Tensor:
#       # Tokenize and chunk the text
#       tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.chunker.chunk_size)
#       input_ids = tokens['input_ids'].squeeze(0)  # Get token IDs
#       chunks = self.chunker.chunk_document(input_ids.tolist())  # Chunk the list of IDs
      
#       # Calculate embeddings for each chunk
#       chunk_embeddings = []
#       for chunk in chunks:
#         chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)  # Decode to text
#         chunk_inputs = self.tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=self.chunker.chunk_size)
#         # Move to GPU if available
#         chunk_inputs = {k: v.to(self.device) for k, v in chunk_inputs.items()}
#         outputs = self.model(**chunk_inputs)
        
#         # Weighted average using attention mask
#         mask = chunk_inputs['attention_mask'].unsqueeze(-1)
#         token_embeddings = outputs.last_hidden_state * mask
#         chunk_embeddings.append(torch.sum(token_embeddings, dim=1) / torch.sum(mask, dim=1))
        
#         return torch.mean(torch.stack(chunk_embeddings), dim=0)  
    
#     def _process_chunk(self, tokens: List[int]) -> torch.Tensor:
#       inputs = self.tokenizer(tokens, return_tensors='pt', max_length=self.chunker.chunk_size, truncation=True)
#       # Move to GPU if available
#       inputs = {k: v.to(self.device) for k, v in inputs.items()}
      
#       outputs = self.model(**inputs)
      
#       # Weighted average using attention mask
#       mask = inputs['attention_mask'].unsqueeze(-1)
#       token_embeddings = outputs.last_hidden_state * mask
      
#       return torch.sum(token_embeddings, dim=1) / torch.sum(mask, dim=1)

class SimilarityCalculator:
    def __init__(self, model_name: str = 'allenai/longformer-base-4096'):
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.eval()
        
        # Modified TF-IDF vectorizer with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Include trigrams
            max_features=100000,  # Increased features
            min_df=2,            # Minimum document frequency
            max_df=0.95,         # Maximum document frequency
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True    # Apply sublinear scaling to term frequencies
        )
        
        self.max_length = self.model.config.max_position_embeddings
        self.chunk_size = 4096
        self.overlap = 256
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.no_grad()
    def calculate_similarity(self, text1: str, text2: str) -> Tuple[float, Dict]:
        # Calculate both similarities
        tfidf_score = self._calculate_tfidf_similarity(text1, text2)
        transformer_score = self._calculate_transformer_similarity(text1, text2)
        
        # Adjust weights based on text length and content
        weight_transformer = min(len(text1), len(text2)) / 8000
        weight_transformer = np.clip(weight_transformer, 0.3, 0.7)

        weight_tfidf = 1 - weight_transformer
        
        # Normalize transformer score
        transformer_score = self._normalize_similarity_score(transformer_score)
        
        # Combined score with dynamic weights
        combined_score = (weight_tfidf * tfidf_score) + (weight_transformer * transformer_score)
        
        return combined_score, {
            "tfidf": tfidf_score,
            "transformer": transformer_score,
            "tfidf_weight": weight_tfidf,
            "transformer_weight": weight_transformer,
            "method": "Combined TF-IDF and Longformer (normalized)"
        }

    def _normalize_similarity_score(self, score: float) -> float:
        return np.clip((score + 1) / 2, 0, 1)  # normalize to [0, 1]


    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        try:
            # Preprocess texts
            texts = [self._preprocess_text(text1), self._preprocess_text(text2)]
            
            # Calculate TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error in TF-IDF calculation: {str(e)}")
            return 0.0

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better similarity comparison
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional, depending on your use case)
        text = re.sub(r'\d+', '', text)
        
        return text.strip()

    def _calculate_transformer_similarity(self, text1: str, text2: str) -> float:
        try:
            # Get embeddings with improved chunking
            emb1 = self._get_document_embedding(text1)
            emb2 = self._get_document_embedding(text2)
            
            # Apply L2 normalization to embeddings
            emb1 = F.normalize(emb1, p=2, dim=-1)
            emb2 = F.normalize(emb2, p=2, dim=-1)
            
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(emb1, emb2, dim=-1).item()
            
            return similarity
        except Exception as e:
            print(f"Error in transformer calculation: {str(e)}")
            return 0.0

    def _get_document_embedding(self, text: str) -> torch.Tensor:
        chunks = self._chunk_text(text)
        chunk_embeddings = []

        for chunk in chunks:
            try:
                # Tokenize with improved padding
                inputs = self.tokenizer(
                    chunk,
                    return_tensors='pt',
                    max_length=self.chunk_size,
                    padding='max_length',
                    truncation=True
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use CLS token embedding instead of average
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # Apply dropout for regularization
                cls_embedding = F.dropout(cls_embedding, p=0.1, training=False)
                
                chunk_embeddings.append(cls_embedding)
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue

        if not chunk_embeddings:
            return torch.zeros((1, self.model.config.hidden_size), device=self.device)

        # Use attention-weighted pooling instead of simple mean
        chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
        attention_weights = torch.softmax(
            torch.matmul(chunk_embeddings, chunk_embeddings.transpose(0, 1)) / 
            np.sqrt(chunk_embeddings.size(-1)), 
            dim=-1
        )
        
        document_embedding = torch.matmul(attention_weights, chunk_embeddings).mean(dim=0, keepdim=True)
        
        return document_embedding

    def _chunk_text(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks, current_chunk = [], []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = current_chunk[-self.overlap:]  # Only retain overlap
                    current_length = len(current_chunk)

            current_chunk.extend(sentence_tokens)
            current_length += sentence_length

        if current_chunk:
            chunks.append(current_chunk)

        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

class ResearchPaperSimilarity:
    def __init__(self):
        self.similarity_calculator = SimilarityCalculator()
    
    def compare_sections(self, sections1: Dict[str, str], sections2: Dict[str, str]) -> Dict[str, Dict]:
        results = {}
        
        for section in sections1.keys():
            if sections1[section] and sections2[section]:
                score, details = self.similarity_calculator.calculate_similarity(
                    sections1[section],
                    sections2[section]
                )
                results[section] = {
                    'similarity_score': score,
                    'details': details
                }
            else:
                results[section] = {
                    'similarity_score': 0.0,
                    'details': {'method': 'No content available'}
                }
                
        return results

def main():
    paper1_path = '../documents/ai content similarity.md'
    paper2_path = '../documents/ai content similarity-2.md'
    
    # Load papers
    get_papers = GetPapers()
    paper1, paper2 = get_papers.load_papers(paper1_path, paper2_path)
    
    if not paper1 or not paper2:
        print("Error loading papers")
        return
    
    # Preprocess papers
    preprocessor = Preprocessor()
    sections_paper1 = preprocessor.extract_sections(paper1)
    sections_paper2 = preprocessor.extract_sections(paper2)
    
    # Initialize detector and compare papers
    detector = ResearchPaperSimilarity()
    results = detector.compare_sections(sections_paper1, sections_paper2)
    
    # Print results
    print('\nSimilarity Scores by Section:')
    print('-' * 50)
    
    for section, result in results.items():
        print(f"\n{section.capitalize()}:")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Method Used: {result['details']['method']}")
        if 'tfidf' in result['details']:
            print(f"TF-IDF Score: {result['details']['tfidf']:.4f}")
            print(f"Transformer Score: {result['details']['transformer']:.4f}")

if __name__ == "__main__":
    main()