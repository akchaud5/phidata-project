import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "research_documents", persist_directory: str = "./data/chroma"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store"""
        try:
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                doc_id = doc.get('id', str(uuid.uuid4()))
                text = doc.get('content', '')
                metadata = {k: v for k, v in doc.items() if k not in ['id', 'content']}
                
                texts.append(text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return []
    
    def search(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False