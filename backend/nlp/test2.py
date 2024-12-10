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
    # def __init__(self, bert_model='allenai/scibert_scivocab_uncased', max_chunk_length=510):
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

    def _calculate_bert_similarity(self, text1, text2):
        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity_score
        except Exception as e:
            print(f"Error in BERT similarity calculation for chunk: {e}")
            return 0.0

    def combined_similarity(self, text1, text2, tfidf_weight=0.3, bert_weight=0.7):
        tfidf_score = self.calculate_tfidf_similarity(text1, text2)
        bert_score = self.calculate_transformer_similarity(text1, text2)
        combined_score = (tfidf_weight * tfidf_score) + (bert_weight * bert_score)
        return combined_score, {"TF-IDF": tfidf_score, "BERT": bert_score}

def main():
    paper1_path = 'C:/College/College Work/plagiarism-detection/backend/documents/research-paper-1.md'
    # paper1_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity.md'
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
            combined_score, individual_scores = similarity_calculator.combined_similarity(text1, text2)
            
            print(f"{section.capitalize():<15} : Combined Score: {combined_score:.4f}")
            print(f"Individual Scores: TF-IDF: {individual_scores['TF-IDF']:.4f}, "
                  f"BERT: {individual_scores['BERT']:.4f}")
        else:
            print(f"{section.capitalize():<15} : No content available")
        print()

if __name__ == "__main__":
    main()
    
    
# class GetPapers:
#     def load_papers(self, paper1_path, paper2_path):
        
#         try:
#             with open(paper1_path, "r", encoding='utf-8') as f1:
#                 paper1_content = f1.read()
#             with open(paper2_path, "r", encoding='utf-8') as f2:
#                 paper2_content = f2.read()
            
#             return paper1_content, paper2_content
        
#         except Exception as e:
#             print(f'Error reading files: {e}')
#             return None, None
        
# class Preprocessor:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
#         self.word_lemmatizer = WordNetLemmatizer()
    
#     def extract_sections(self, paper_content):
#         sections = {
#             'abstract': '',
#             'introduction': '',
#             'methodology': '',
#             'results': '',
#             'discussion': '',
#             'conclusion': '',
#             'full_text': paper_content
#         }
        
#         section_patterns = {
#             'abstract': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:abstract).*?(?=\n#+|$)',
#             'introduction': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:introduction|background).*?(?=\n#+|$)',
#             'methodology': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:methodology|methods|materials and methods).*?(?=\n#+|$)',
#             'results': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:results|findings).*?(?=\n#+|$)',
#             'discussion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:discussion).*?(?=\n#+|$)',
#             'conclusion': r'(?i)(?:^|\n)#+\s*(?:(?:[IVX]+\.?\s+)?)?(?:conclusion|final thoughts).*?(?=\n#+|$)'
#         }
        
#         for section, pattern in section_patterns.items():
#             match = re.search(pattern, paper_content, re.DOTALL)
#             if match:
#                 content = re.sub(r'^#+\s*\w+\s*', '', match.group(0), flags=re.IGNORECASE)
#                 sections[section] = content.strip()
                
#         return sections
    
#     def preprocess_text(self, text, use_lemmatization=True):
#         if not text:
#             return ''
        
#         try:
#             text = re.sub(r'[^a-zA-Z0-9\s.,]', ' ', text)
#             text = text.lower()
            
#             tokens = nltk.word_tokenize(text)
#             filtered_tokens = [
#                 self.word_lemmatizer.lemmatize(token) if use_lemmatization else token
#                 for token in tokens if token not in self.stop_words
#             ]
            
#             return ' '.join(filtered_tokens)
            
#         except Exception as e:
#             print(f"Error in preprocessing: {str(e)}")
#             return ''
   
# class SimilarityCalculator:
#     def __init__(self, bert_model='all-MiniLM-L6-v2', longformer_model='allenai/longformer-base-4096'):
#         self.bert_model = SentenceTransformer(bert_model)
#         self.longformer_calculator = LongformerSimilarityCalculator(longformer_model)
#         self.bert_max_tokens = 512  # BERT's max token limit
#         self.tokenizer = LongformerTokenizer.from_pretrained(longformer_model)
#         self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))

#     def calculate_tfidf_similarity(self, text1, text2):
#         if not text1 or not text2:
#             return 0.0
#         try:
#             tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
#             return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#         except Exception as e:
#             print(f"Error in TF-IDF calculation: {str(e)}")
#             return 0.0

#     def calculate_bert_similarity(self, text1, text2):
#         try:
#             embeddings = self.bert_model.encode([text1, text2])
#             return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#         except Exception as e:
#             print(f"Error in BERT calculation: {str(e)}")
#             return 0.0

#     def calculate_similarity(self, text1, text2):
#         # Check the token count of both texts
#         tokens1 = self.tokenizer.encode(text1)
#         tokens2 = self.tokenizer.encode(text2)
        
#         # If either text exceeds BERT's limit, use Longformer
#         if len(tokens1) > self.bert_max_tokens or len(tokens2) > self.bert_max_tokens:
#             print(f"Using Longformer for comparison (token counts: {len(tokens1)}, {len(tokens2)})")
#             return self.longformer_calculator.calculate_similarity(text1, text2)
#         else:
#             print(f"Using BERT for comparison (token counts: {len(tokens1)}, {len(tokens2)})")
#             return self.calculate_bert_similarity(text1, text2)
    
# class ResearchPaperSimilarityDetector:
#     def __init__(self, tfidf_weight=0.3, bert_weight=0.7):
#         self.similarity_calculator = SimilarityCalculator()
#         self.tfidf_weight = tfidf_weight
#         self.bert_weight = bert_weight
        
#     def combined_similarity(self, text1, text2):
#         tfidf_score = self.similarity_calculator.calculate_tfidf_similarity(text1, text2)
        
#         try:
#             bert_score = self.similarity_calculator.calculate_similarity(text1, text2)
#             model_used = self.get_model_used(text1, text2)
#         except Exception as e:
#             print(f"Error in transformer calculation: {str(e)}. Falling back to TF-IDF only.")
#             bert_score = 0
#             model_used = "Fallback to TF-IDF"
        
#         if bert_score == 0 and model_used != "Fallback to TF-IDF":
#             print(f"Transformer score is 0. Attempting to use the other transformer model.")
#             try:
#                 if model_used == "BERT":
#                     bert_score = self.similarity_calculator.longformer_calculator.calculate_similarity(text1, text2)
#                     model_used = "Longformer (Fallback)"
#                 else:
#                     bert_score = self.similarity_calculator.calculate_bert_similarity(text1, text2)
#                     model_used = "BERT (Fallback)"
#             except Exception as e:
#                 print(f"Error in fallback transformer calculation: {str(e)}. Using TF-IDF only.")
#                 model_used = "Fallback to TF-IDF"
        
#         if bert_score == 0:
#             combined_score = tfidf_score
#         else:
#             combined_score = (self.tfidf_weight * tfidf_score) + (self.bert_weight * bert_score)
        
#         return combined_score, {
#             "tfidf": tfidf_score,
#             "transformer": bert_score
#         }, model_used
    
#     def get_model_used(self, text1, text2):
#         tokens1 = self.similarity_calculator.tokenizer.encode(text1)
#         tokens2 = self.similarity_calculator.tokenizer.encode(text2)
        
#         if len(tokens1) > self.similarity_calculator.bert_max_tokens or len(tokens2) > self.similarity_calculator.bert_max_tokens:
#             return "Longformer"
#         else:
#             return "BERT"
    
#     def auto_combined_similarity(self, text1, text2):
#         tokens1 = self.similarity_calculator.tokenizer.encode(text1)
#         tokens2 = self.similarity_calculator.tokenizer.encode(text2)
        
#         if len(tokens1) > self.similarity_calculator.bert_max_tokens or len(tokens2) > self.similarity_calculator.bert_max_tokens:
#             print(f"Using Longformer for comparison (token counts: {len(tokens1)}, {len(tokens2)})")
#             return self.combined_similarity_longformer(text1, text2)
#         else:
#             print(f"Using BERT for comparison (token counts: {len(tokens1)}, {len(tokens2)})")
#             return self.combined_similarity_bert(text1, text2)

# class LongformerSimilarityCalculator:
#     def __init__(self, model_name='allenai/longformer-base-4096'):
#         self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
#         self.model = LongformerModel.from_pretrained(model_name)
#         self.max_length = self.model.config.max_position_embeddings

#     def calculate_similarity(self, text1, text2):
#         try:
#             # Tokenize and truncate
#             inputs1 = self.tokenizer(text1, return_tensors='pt', max_length=self.max_length, truncation=True)
#             inputs2 = self.tokenizer(text2, return_tensors='pt', max_length=self.max_length, truncation=True)
            
#             with torch.no_grad():
#                 outputs1 = self.model(**inputs1)
#                 outputs2 = self.model(**inputs2)
            
#             # Use mean pooling instead of just the [CLS] token
#             mask1 = inputs1['attention_mask'].unsqueeze(-1).expand(outputs1.last_hidden_state.size()).float()
#             mask2 = inputs2['attention_mask'].unsqueeze(-1).expand(outputs2.last_hidden_state.size()).float()
            
#             sum_embeddings1 = torch.sum(outputs1.last_hidden_state * mask1, 1)
#             sum_embeddings2 = torch.sum(outputs2.last_hidden_state * mask2, 1)
            
#             sum_mask1 = torch.clamp(mask1.sum(1), min=1e-9)
#             sum_mask2 = torch.clamp(mask2.sum(1), min=1e-9)
            
#             embedding1 = sum_embeddings1 / sum_mask1
#             embedding2 = sum_embeddings2 / sum_mask2
            
#             similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())[0][0]
#             return similarity
#         except Exception as e:
#             print(f"Error in Longformer calculation: {str(e)}")
#             return 0.0

# def main():
#     paper1_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity.md'
#     paper2_path = 'C:/College/College Work/plagiarism-detection/backend/documents/ai content similarity-2.md'
    
#     get_papers = GetPapers()
#     paper1, paper2 = get_papers.load_papers(paper1_path, paper2_path)
    
#     if not paper1 or not paper2:
#         print("Error loading papers")
#         return
    
#     preprocessor = Preprocessor()
#     sections_paper1 = preprocessor.extract_sections(paper1)
#     sections_paper2 = preprocessor.extract_sections(paper2)
    
#     if not sections_paper1 or not sections_paper2:
#         print("Error extracting sections")
#         return
    
#     similarity_detector = ResearchPaperSimilarityDetector(tfidf_weight=0.3, bert_weight=0.7)
    
#     print('Similarity Scores by Section:')
#     print('-' * 30)
    
#     for section in sections_paper1.keys():
#         text1 = preprocessor.preprocess_text(sections_paper1[section])
#         text2 = preprocessor.preprocess_text(sections_paper2[section])
        
#         if text1 and text2:  # Only process if both texts have content
#             combined_score, individual_scores, model_used = similarity_detector.combined_similarity(text1, text2)
            
#             print(f"{section.capitalize():<15} : Combined Score: {combined_score:.4f}")
#             print(f"Model used: {model_used}")
#             print(f"Individual Scores: TF-IDF: {individual_scores['tfidf']:.4f}, "
#                   f"Transformer ({model_used}): {individual_scores['transformer']:.4f}")
#         else:
#             print(f"{section.capitalize():<15} : No content available")
#         print()  # Add a blank line for readability

# if __name__ == "__main__":
#     main()
