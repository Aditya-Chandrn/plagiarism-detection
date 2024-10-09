import re
import nltk
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer

class ResearchPaperDetector:
    def __init__(self, bert_weight=0.6):
        """
        Initializes the detector with a combination of BERT embeddings and TF-IDF.
        The bert_weight parameter determines the importance of BERT embeddings in the similarity score.
        """
        # Set the BERT and TF-IDF weighting
        self.bert_weight = bert_weight
        self.tfidf_weight = 1 - bert_weight
        
        # Vectorizers for different sections of the research paper (TF-IDF)
        self.section_vectorizers = {
            'abstract': TfidfVectorizer(ngram_range=(1, 3)),
            'introduction': TfidfVectorizer(ngram_range=(1, 3)),
            'methodology': TfidfVectorizer(ngram_range=(1, 3)),
            'results': TfidfVectorizer(ngram_range=(1, 3)),
            'discussion': TfidfVectorizer(ngram_range=(1, 3))
        }
        
        # Initialize the BERT model for sentence embeddings
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load stopwords and lemmatizer for text preprocessing
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Regular expressions for detecting and removing citation patterns
        self.citation_patterns = [
            r'\(\w+,?\s+\d{4}\)',  # (Author, 2020)
            r'\[[\d,\s-]+\]',      # [1], [1,2,3]
            r'\d+\.\s+references', # For removing reference lists
            r'(?<=\])\d{1,3}(?=[,\s])', # Numbered citations
        ]

    def extract_paper_sections(self, text):
        """
        Extracts the different sections (abstract, introduction, etc.) of a research paper.
        Returns a dictionary with sections and their content.
        """
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'full_text': text  # Keep the entire paper text
        }
        
        # Patterns to detect common section headers
        section_patterns = {
            'abstract': r'abstract.*?(?=introduction|\n\n)',
            'introduction': r'introduction.*?(?=methodology|methods|materials and methods|\n\n)',
            'methodology': r'(methodology|methods|materials and methods).*?(?=results|\n\n)',
            'results': r'results.*?(?=discussion|\n\n)',
            'discussion': r'discussion.*?(?=conclusion|references|\n\n)'
        }
        
        # Search for each section using regex and store in sections dictionary
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            if match:
                sections[section] = match.group(0)
        
        return sections

    def preprocess_academic_text(self, text):
        """
        Preprocesses academic text by removing citations and common phrases,
        tokenizing, lemmatizing, and removing stopwords.
        """
        # Remove citations using predefined citation patterns
        for pattern in self.citation_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove common academic phrases that don’t contribute to similarity detection
        common_phrases = [
            'in this paper', 'in this study', 'in this research',
            'the results show', 'the findings indicate',
            'according to the results', 'based on the findings'
        ]
        for phrase in common_phrases:
            text = text.replace(phrase.lower(), '')
        
        # Tokenize the text and apply POS tagging
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        
        # Lemmatize tokens and remove stopwords
        processed_tokens = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for word, tag in tagged_tokens
            if word.lower() not in self.stop_words and word.isalpha()
        ]
        
        return ' '.join(processed_tokens)

    def get_wordnet_pos(self, nltk_tag):
        """
        Maps NLTK part-of-speech tags to WordNet format for lemmatization.
        """
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, 
                   "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(nltk_tag[0].upper(), wordnet.NOUN)

    def calculate_section_similarity(self, section1, section2, section_name):
        """
        Calculates similarity between two sections using a combination of TF-IDF and BERT embeddings.
        """
        if not section1 or not section2:
            return 0.0
        
        # Calculate TF-IDF similarity for the section
        vectorizer = self.section_vectorizers[section_name]
        tfidf_matrix = vectorizer.fit_transform([section1, section2])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Calculate BERT similarity for the section
        bert_embeddings = self.bert_model.encode([section1, section2])
        bert_similarity = cosine_similarity([bert_embeddings[0]], [bert_embeddings[1]])[0][0]
        
        # Return a weighted combination of TF-IDF and BERT similarities
        return (self.bert_weight * bert_similarity + 
                self.tfidf_weight * tfidf_similarity)

    def analyze_paper_similarity(self, pdf1_path, pdf2_path):
        """
        Compares two research papers by extracting their sections and calculating similarity.
        Returns an analysis report with section-wise and overall similarity scores.
        """
        # Extract text from the two PDF files
        paper1 = self.extract_text_from_pdf(pdf1_path)
        paper2 = self.extract_text_from_pdf(pdf2_path)
        
        # Extract sections from each paper
        sections1 = self.extract_paper_sections(paper1)
        sections2 = self.extract_paper_sections(paper2)
        
        # Section-wise similarity scores
        section_scores = {}
        section_weights = {
            'abstract': 0.15,
            'introduction': 0.2,
            'methodology': 0.25,
            'results': 0.25,
            'discussion': 0.15
        }
        
        # Calculate similarity for each section
        for section in sections1.keys():
            if section != 'full_text':
                processed_section1 = self.preprocess_academic_text(sections1[section])
                processed_section2 = self.preprocess_academic_text(sections2[section])
                section_scores[section] = self.calculate_section_similarity(
                    processed_section1, processed_section2, section
                )
        
        # Calculate weighted overall similarity
        weighted_similarity = sum(
            section_scores[section] * section_weights[section]
            for section in section_scores.keys()
        )
        
        # Generate a detailed analysis report
        return self.generate_analysis_report(section_scores, weighted_similarity)

    def generate_analysis_report(self, section_scores, weighted_similarity):
        """
        Generates a detailed analysis report based on section scores and overall similarity.
        """
        report = {
            'overall_similarity': weighted_similarity,
            'section_scores': section_scores,
            'interpretation': self.interpret_academic_similarity(weighted_similarity),
            'risk_level': self.assess_risk_level(weighted_similarity, section_scores)
        }
        
        return report

    def interpret_academic_similarity(self, similarity):
        """
        Provides an interpretation of the overall similarity score.
        """
        if similarity >= 0.8:
            return ("HIGH SIMILARITY - Significant overlap detected. "
                   "Thorough review recommended for potential duplication.")
        elif similarity >= 0.6:
            return ("MODERATE SIMILARITY - Some overlapping content. "
                   "May indicate similar methodology or common background material.")
        else:
            return ("LOW SIMILARITY - Likely original research. "
                   "Common academic phrases may account for minor similarities.")

    def assess_risk_level(self, overall_similarity, section_scores):
        """
        Assesses the risk level of potential plagiarism based on section-wise and overall similarity.
        """
        risk_factors = []
        
        # High similarity in key sections increases risk
        if section_scores.get('methodology', 0) > 0.8:
            risk_factors.append("High methodology similarity")
        if section_scores.get('results', 0) > 0.7:
            risk_factors.append("High results similarity")
            
        # Final risk assessment based on overall similarity and risk factors
        if overall_similarity > 0.8 or len(risk_factors) >= 2:
            return "High Risk"
        elif overall_similarity > 0.6 or len(risk_factors) == 1:
            return "Medium Risk"
        return "Low Risk"

    def extract_text_from_pdf(self, file_path):
        """
        Extracts text from a PDF file.
        """
        text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return ' '.join(text)

# Example usage
if __name__ == "__main__":
    detector = ResearchPaperDetector(bert_weight=0.6)
    
    # Compare papers
    paper1_path = "documents/research-paper-1.pdf"
    paper2_path = "documents/research-paper-2.pdf"
    
    results = detector.analyze_paper_similarity(paper1_path, paper2_path)
    
    # Print detailed results
    print("\nResearch Paper Similarity Analysis:")
    print(f"\nOverall Similarity Score: {results['overall_similarity']:.4f}")
    print(f"Risk Level: {results['risk_level']}")
    print(f"\nInterpretation: {results['interpretation']}")
    
    print("\nSection-wise Similarity Scores:")
    for section, score in results['section_scores'].items():
        print(f"{section.title()}: {score:.4f}")
