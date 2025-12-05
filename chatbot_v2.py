"""
Enhanced FAQ-Style RAG Chatbot with Groq API
Strategy: Semantic search on questions → Retrieve exact answers → LLM reformulation
"""
import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv()


class SorosAdvisorV2:
    """FAQ-style RAG: Match question semantically, return exact answer from Excel"""
    
    CONFIDENCE_THRESHOLD = 0.60
    MAX_QUERY_LENGTH = 500
    
    def __init__(self, index_name: str = "soros-chatbot", use_groq: bool = True):
        self.vector_store = VectorStore(index_name=index_name)
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY")
        self.llm_client = None
        
        if self.use_groq:
            self._init_groq()
        
        # Cache for category classification
        self.categories = ["Personal Life", "Investment Strategy", "Trading Philosophy", 
                          "Risk Management", "Market Analysis", "Financial History"]
    
    def _init_groq(self):
        """Initialize Groq API client"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your-groq-api-key-here":
                print("⚠️  GROQ_API_KEY not set. Using direct answers without LLM reformulation.")
                self.use_groq = False
                return
            
            self.llm_client = Groq(api_key=api_key)
            print("✅ Groq API initialized (Llama 3.1 70B)")
        except ImportError:
            print("⚠️  groq package not installed. Run: pip install groq")
            self.use_groq = False
        except Exception as e:
            print(f"⚠️  Groq initialization failed: {e}")
            self.use_groq = False
    
    def _validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate user query"""
        if not query or not query.strip():
            return False, "Please ask a question."
        
        if len(query) > self.MAX_QUERY_LENGTH:
            return False, f"Question too long. Maximum {self.MAX_QUERY_LENGTH} characters."
        
        # Check if it's Soros-related (basic check)
        soros_keywords = ['soros', 'george', 'invest', 'trade', 'market', 'stock', 
                         'strategy', 'portfolio', 'risk', 'fund', 'money', 'wealth']
        query_lower = query.lower()
        
        if not any(keyword in query_lower for keyword in soros_keywords):
            # Allow general questions, but warn
            pass
        
        return True, None
    
    def _classify_category(self, query: str) -> Optional[str]:
        """Optional: Classify query category for pre-filtering (future enhancement)"""
        # For now, skip classification and search all categories
        # Can add lightweight classifier here later
        return None
    
    def chat(self, query: str) -> Dict:
        """
        Main chat method - FAQ-style RAG
        1. Validate query
        2. Semantic search on QUESTIONS (not answers)
        3. Retrieve exact ANSWER from matched Q&A pair
        4. Use LLM to reformulate answer conversationally
        """
        # Step 1: Validate
        is_valid, error_msg = self._validate_query(query)
        if not is_valid:
            return {
                "response": error_msg,
                "confidence": 0.0,
                "in_db": False,
                "sources": [],
                "method": "validation_error"
            }
        
        # Step 2: Semantic search on questions
        results = self.vector_store.search(query, top_k=5)
        
        if not results:
            return {
                "response": "I couldn't find relevant information in George Soros's knowledge base. Could you rephrase your question?",
                "confidence": 0.0,
                "in_db": False,
                "sources": [],
                "method": "no_results"
            }
        
        # Get top match (results are tuples: (text, score))
        matched_text, confidence = results[0]
        
        # Step 3: Extract the exact answer from the Q&A pair
        # Format in vector store: "Question: ... Answer: ..."
        exact_answer = self._extract_answer(matched_text)
        
        # Step 4: Check confidence threshold
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "response": f"I found some information but it's not very relevant (confidence: {confidence*100:.2f}%). Could you rephrase your question?",
                "confidence": confidence,
                "in_db": False,
                "sources": [matched_text],
                "method": "low_confidence"
            }
        
        # Step 5: Use LLM to reformulate answer (if Groq available)
        if self.use_groq and self.llm_client:
            # Convert results to dict format for reformulation
            context_results = [{"text": text, "score": score} for text, score in results[:3]]
            reformulated_answer = self._reformulate_with_groq(query, exact_answer, context_results)
            method = "groq_reformulation"
        else:
            # Fallback: return exact answer
            reformulated_answer = exact_answer
            method = "direct_answer"
        
        return {
            "response": reformulated_answer,
            "confidence": confidence,
            "in_db": True,
            "sources": [text for text, score in results[:3]],
            "method": method,
            "exact_answer": exact_answer  # Keep original for comparison
        }
    
    def _extract_answer(self, qa_text: str) -> str:
        """Extract answer portion from 'Question: ... Answer: ...' format"""
        if "Answer:" in qa_text:
            parts = qa_text.split("Answer:", 1)
            return parts[1].strip()
        return qa_text  # Fallback if format unexpected
    
    def _batch_upsert(self, questions: List[str], qa_texts: List[str], metadatas: List[dict], batch_size: int = 100):
        """
        Custom batch upsert that vectorizes QUESTIONS but stores full Q&A text
        """
        print(f"Generating embeddings for {len(questions)} questions...")
        embeddings = self.vector_store.embedding_model.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        print(f"Uploading to Pinecone...")
        vectors = []
        for i, (question, qa_text, embedding, metadata) in enumerate(zip(questions, qa_texts, embeddings, metadatas)):
            meta = {"text": qa_text}  # Store full Q&A text
            meta.update(metadata)  # Add extra metadata
            
            vectors.append({
                "id": f"doc_{i}",
                "values": embedding.tolist(),
                "metadata": meta
            })
            
            if len(vectors) >= batch_size:
                self.vector_store.index.upsert(vectors=vectors)
                vectors = []
        
        if vectors:
            self.vector_store.index.upsert(vectors=vectors)
        
        print(f"✅ Successfully uploaded {len(questions)} Q&A pairs")
    
    def _reformulate_with_groq(self, user_query: str, exact_answer: str, context_results: List[Dict]) -> str:
        """
        Use Groq API to reformulate the exact answer conversationally
        Uses multiple context results to provide comprehensive answers
        """
        try:
            # Extract all matched questions and answers
            matched_qa_pairs = []
            for i, r in enumerate(context_results[:3], 1):
                qa_text = r['text']
                if "Answer:" in qa_text:
                    q_part, a_part = qa_text.split("Answer:", 1)
                    question = q_part.replace("Question:", "").strip()
                    answer = a_part.strip()
                    matched_qa_pairs.append(f"Q{i}: {question}\nA{i}: {answer}")
            
            all_context = "\n\n".join(matched_qa_pairs)
            
            # Get the primary matched question
            matched_question = context_results[0]['text'].split("Answer:")[0].replace("Question:", "").strip()
            
            # Intelligent prompt that handles both specific and general questions
            system_prompt = """You are George Soros's AI investment advisor. 
You provide helpful, accurate answers using ONLY information from the database.

CRITICAL RULES:
1. Answer DIRECTLY and conversationally - do NOT explain your reasoning process
2. For GENERAL questions (e.g., "Tell me about X"): Combine facts from multiple Q&A pairs into a comprehensive 3-5 sentence response
3. For SPECIFIC questions (e.g., "When/What/How did X happen?"): Give a focused 1-3 sentence answer
4. ENTITY CHECK: If the Q&A is about a DIFFERENT person/entity, say "I don't have that information in my database"
5. NEVER add facts not in the provided Q&A pairs
6. Be natural, conversational, and direct - like a knowledgeable advisor
7. Do NOT start with phrases like "The user's question is..." or "According to..."
8. Jump straight into the answer"""

            user_prompt = f"""User Question: {user_query}

Available Information:
{all_context}

Provide a direct, conversational answer to the user's question using ONLY the facts above. Do not explain your reasoning - just answer naturally."""

            # Call Groq API
            response = self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature = less creativity, more accuracy
                max_tokens=300,  # Increased for general questions
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fallback to exact answer
            return exact_answer
    
    def train(self, excel_path: str) -> Dict:
        """
        Train the FAQ-style RAG by vectorizing QUESTIONS from Excel
        
        Strategy:
        1. Load Excel with Question/Answer columns
        2. Create embeddings for QUESTIONS only
        3. Store Q&A pairs with metadata in Pinecone
        """
        try:
            df = pd.read_excel(excel_path)
            
            # Flexible column detection
            question_col = None
            answer_col = None
            label_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'question' in col_lower and not question_col:
                    question_col = col
                elif 'answer' in col_lower and not answer_col:
                    answer_col = col
                elif 'label' in col_lower or 'category' in col_lower:
                    label_col = col
            
            if not question_col or not answer_col:
                return {"error": "Excel must have 'Question' and 'Answer' columns"}
            
            print(f"✅ Found columns: {question_col}, {answer_col}")
            if label_col:
                print(f"✅ Found category column: {label_col}")
            
            # Prepare Q&A pairs
            qa_pairs = []
            for idx, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()
                
                # Skip empty rows
                if not question or not answer or question == 'nan' or answer == 'nan':
                    continue
                
                # Format for storage: This is what goes into vector DB
                # We vectorize the QUESTION, but store the full Q&A pair
                qa_text = f"Question: {question}\nAnswer: {answer}"
                
                metadata = {
                    "question": question,
                    "answer": answer,
                    "row_index": idx
                }
                
                if label_col and pd.notna(row.get(label_col)):
                    metadata["category"] = str(row[label_col])
                
                qa_pairs.append((question, qa_text, metadata))  # Vectorize question, store full Q&A
            
            print(f"📊 Processing {len(qa_pairs)} Q&A pairs...")
            
            # Prepare for vectorization
            # We vectorize QUESTIONS but store full Q&A text in metadata
            questions_to_vectorize = []
            qa_texts_to_store = []
            metadatas = []
            
            for question, qa_text, metadata in qa_pairs:
                questions_to_vectorize.append(question)  # This gets vectorized
                qa_texts_to_store.append(qa_text)  # This gets stored in metadata as "text"
                metadatas.append(metadata)  # Additional metadata
            
            # Delete old data to avoid conflicts
            print("🗑️  Clearing old vectors...")
            try:
                self.vector_store.index.delete(delete_all=True)
                time.sleep(2)  # Wait for deletion to complete
            except Exception as e:
                print(f"⚠️  Skip deletion (index empty): {e}")
                # Continue anyway - upsert will overwrite
            
            # Upload to Pinecone with proper structure
            # add_documents will:
            # 1. Vectorize the questions
            # 2. Store the full Q&A text in metadata["text"]
            # 3. Add additional metadata (category, row_index, etc.)
            print(f"📤 Uploading {len(qa_pairs)} vectors to Pinecone...")
            self._batch_upsert(questions_to_vectorize, qa_texts_to_store, metadatas)
            
            return {
                "success": True,
                "total_pairs": len(qa_pairs),
                "message": f"Successfully indexed {len(qa_pairs)} Q&A pairs"
            }
            
        except Exception as e:
            return {"error": str(e)}


# Quick test
if __name__ == "__main__":
    chatbot = SorosAdvisorV2()
    
    # Test query
    result = chatbot.chat("What is George Soros's net worth?")
    print("\n" + "="*60)
    print(f"Question: What is George Soros's net worth?")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Method: {result['method']}")
    print(f"\nAnswer: {result['response']}")
    print("="*60)
