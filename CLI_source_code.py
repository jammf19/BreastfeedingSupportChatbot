#import libraries
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the CSV data with error handling
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'preprocessed_lll_data.csv')

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found at {csv_path}. Please ensure the file exists and the path is correct.")

try:
    df = pd.read_csv(csv_path)
    paragraphs = df['paragraph'].tolist()
    paragraph_sources = df['source'].tolist()
    post_titles = df['title'].tolist()
except Exception as e:
    raise RuntimeError(f"Failed to load data from {csv_path}: {e}")


class LaLecheLeagueChatbot:
    def __init__(self, csv_path):
        # Load the CSV data
        self.df = pd.read_csv(csv_path)
        
        # Initialize the models
        self.initialize_models()
        
        # Process the documents
        self.process_documents()
        
    def initialize_models(self):
        # 1. Sentence-BERT model for encoding text into embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Question-answering pipeline for extracting specific answers
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        
        # 3. Summarization pipeline for condensing long answers
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            max_length=150,
            min_length=40,
            use_auth_token=True
        )
        
        # 4. Intent classification for detecting query types
        self.intent_classifier = pipeline(
            "text-classification",
            model="typeform/distilbert-base-uncased-mnli"
        )
        
    def process_documents(self):
        # Extract paragraphs from content to make chunks
        self.paragraphs = []
        self.paragraph_sources = []
        self.post_titles = []
        
        for _, row in self.df.iterrows():
            content = row['paragraph']
            title = row['title']
            url = row['source']
            
            # Split content into paragraphs
            content_paragraphs = content.split('\n')
            
            for para in content_paragraphs:
                # Skip empty paragraphs or very short ones
                if len(para.strip()) < 20:
                    continue
                    
                self.paragraphs.append(para)
                self.paragraph_sources.append(url)
                self.post_titles.append(title)
        
        # Create embeddings for each paragraph
        self.paragraph_embeddings = self.embedding_model.encode(self.paragraphs)
        
    def find_relevant_content(self, query, top_k=3):
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            [query_embedding], 
            self.paragraph_embeddings
        )[0]
        
        # Get indices of top K most similar paragraphs
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_content = []
        for idx in top_indices:
            relevant_content.append({
                'paragraph': self.paragraphs[idx],
                'source': self.paragraph_sources[idx],
                'title': self.post_titles[idx],
                'score': similarities[idx]
            })
            
        return relevant_content
    
    def answer_question(self, query):
        # First, find relevant content
        relevant_content = self.find_relevant_content(query, top_k=3)
        
        if not relevant_content:
            return {
                'answer': "I'm sorry, I couldn't find information about that in the La Leche League resources.",
                'source': None,
                'context': None
            }
        
        # Combine the most relevant paragraphs into a context
        context = " ".join([item['paragraph'] for item in relevant_content])
        
        # Use question-answering to extract a specific answer
        try:
            qa_result = self.qa_model(
                question=query,
                context=context
            )
            
            answer = qa_result['answer']
            score = qa_result['score']
            
            # If confidence is low, use the whole paragraph
            if score < 0.1:
                answer = relevant_content[0]['paragraph']
                
            return {
                'answer': answer,
                'source': relevant_content[0]['source'],
                'title': relevant_content[0]['title'],
                'context': context
            }
            
        except Exception as e:
            # Fallback to the most relevant paragraph
            return {
                'answer': relevant_content[0]['paragraph'],
                'source': relevant_content[0]['source'],
                'title': relevant_content[0]['title'],
                'context': context
            }
        
    def detect_intent(self, query):
        """Detect the user's intent"""
        # Map common intents
        result = self.intent_classifier(
            query, 
            candidate_labels=["question", "instruction", "personal story", "help request"]
        )
        
        return result['labels'][0]
    
    def summarize_content(self, text):
        """Summarize long content"""
        # Only summarize if text is long enough
        if len(text.split()) < 100:
            return text
            
        try:
            summary = self.summarizer(text)
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback to original text if summarization fails
            return text
    
    def get_response(self, user_input):
        """Enhanced main method to interact with the chatbot"""
        # Get the answer using the retrieval-based approach
        result = self.answer_question(user_input)
        
        # Consider summarizing if the answer is long
        answer_text = result['answer']
        if len(answer_text.split()) > 100:
            answer_text = self.summarize_content(answer_text)
        
        # Build a more informative response
        response = f"{answer_text}\n\n"

        response += f"For more information please view the La Leche League article: '{result['title']}'\n"
        response += f"Source: {result['source']}"
        
        # Add related topics suggestion based on the source article
        related_content = self.find_relevant_content(result['title'], top_k=2)
        if related_content and len(related_content) > 1:
            related_titles = set([content['title'] for content in related_content 
                               if content['title'] != result['title']])
            if related_titles:
                response += f"\n\nYou might also be interested in: {', '.join(related_titles)} \n\nsource: {related_content[0]['source']}"
        
        return response

# Initialize the chatbot
chatbot = LaLecheLeagueChatbot(csv_path)
    
# Interactive loop
print("Breastfeeding Information Assistant (type 'exit' to quit)")
print("Please ask your question regarding breastfeeding:")
    
while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        break
            
    response = chatbot.get_response(user_input)
    print("\n" + response + "\n")       