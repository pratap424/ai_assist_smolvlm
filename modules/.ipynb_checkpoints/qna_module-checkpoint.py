from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class QnASystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.context = ""
        self.chunks = []
        self.embeddings = None
        
    def update_context(self, context):
        """Update the knowledge base with new context"""
        self.context = context
        self._process_context()
        
    def _process_context(self):
        """Process context into chunks with embeddings"""
        # Split context into meaningful chunks
        self.chunks = self._split_into_chunks(self.context)
        
        # Generate embeddings for each chunk
        if self.chunks:
            self.embeddings = self.model.encode(self.chunks)
            
    def _split_into_chunks(self, text, max_length=256):
        """Split text into manageable chunks"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def answer_question(self, question, top_n=3):
        """Answer question based on stored context"""
        if not self.chunks or not self.embeddings.any():
            return "No context available to answer questions."
            
        # Encode question
        question_embedding = self.model.encode([question])
        
        # Calculate similarity scores
        scores = cosine_similarity(question_embedding, self.embeddings)[0]
        
        # Get top N relevant chunks
        top_indices = np.argsort(scores)[-top_n:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices]
        
        # Generate answer (simple version - could be enhanced with LLM)
        answer = self._generate_answer(question, relevant_chunks)
        return answer
        
    def _generate_answer(self, question, contexts):
        """Simple answer generation from context chunks"""
        context_str = "\n".join(contexts)
        
        # Simple template-based generation
        return f"Based on the context:\n{context_str}\n\nAnswer to '{question}': " + \
               "Here are the relevant details from the scene: " + \
               " ".join(contexts)[:500] + "..."

    def clear_context(self):
        """Clear existing context"""
        self.context = ""
        self.chunks = []
        self.embeddings = None