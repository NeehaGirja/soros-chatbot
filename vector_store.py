"""
Pinecone cloud vector database
"""
import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv()


class VectorStore:
    def __init__(self, index_name: str = "soros-chatbot", dimension: int = 384):
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = dimension
        self.index_name = index_name
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found. Create .env file with your API key.")
        
        self.pc = Pinecone(api_key=api_key)
        self._setup_index()
        
    def _setup_index(self):
        """Setup Pinecone index with cosine similarity for efficient semantic search"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            # Cosine metric: Efficient for normalized embeddings (sentence transformers auto-normalize)
            # Serverless: Auto-scaling, cost-efficient for 905 vectors
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Best for semantic similarity
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Low latency for US
                )
            )
            time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        
    def add_documents(self, documents: List[str], batch_size: int = 100, metadatas: List[dict] = None):
        """
        Efficient batch upload to Pinecone
        - batch_size=100: Optimal for Pinecone serverless
        - Embeddings cached by sentence-transformers
        - metadatas: Optional list of metadata dicts for each document
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        # GPU acceleration if available, otherwise CPU
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Optimize memory usage
        )
        
        print(f"Uploading to Pinecone...")
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = {"text": doc}  # Store original Q&A for retrieval
            if metadatas and i < len(metadatas):
                metadata.update(metadatas[i])
            
            vectors.append({
                "id": f"doc_{i}",
                "values": embedding.tolist(),
                "metadata": metadata
            })
            
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
        
        if vectors:
            self.index.upsert(vectors=vectors)
        
        print(f"Successfully uploaded {len(documents)} documents")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Semantic search using cosine similarity
        Returns: [(document, similarity_score), ...]
        Score range: 0-1 (higher = more similar)
        """
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Pinecone query optimized for low latency
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True  # Return original text
        )
        
        return [(match['metadata'].get('text', ''), match['score']) for match in results['matches']]
    
    def get_stats(self) -> dict:
        """Check index efficiency: vector count, dimension, etc."""
        return self.index.describe_index_stats()
    
    def __len__(self):
        stats = self.get_stats()
        return stats.get('total_vector_count', 0)