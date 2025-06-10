import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.documents = []
        self.embeddings = None
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search index"""
        self.documents.extend(documents)
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the search index with current documents"""
        if not self.documents:
            return
        
        try:
            # Extract text content for indexing
            texts = [self._extract_searchable_text(doc) for doc in self.documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Generate TF-IDF vectors
            logger.info("Building TF-IDF index...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            self.is_fitted = True
            logger.info(f"Search index built with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building search index: {e}")
            self.is_fitted = False
    
    def _extract_searchable_text(self, doc: Dict[str, Any]) -> str:
        """Extract searchable text from document"""
        text_parts = []
        
        # Add title with higher weight
        title = doc.get('title', '')
        if title:
            text_parts.append(f"{title} {title}")  # Duplicate for weight
        
        # Add content
        content = doc.get('content', '')
        if content:
            text_parts.append(content)
        
        # Add metadata fields
        metadata = doc.get('metadata', {})
        
        # Add authors
        authors = metadata.get('authors', [])
        if authors:
            text_parts.append(" ".join(authors))
        
        # Add categories/topics
        categories = metadata.get('categories', [])
        topics = metadata.get('topics', [])
        all_categories = categories + topics
        if all_categories:
            text_parts.append(" ".join(all_categories))
        
        # Add abstract/summary/description
        for field in ['abstract', 'summary', 'description']:
            if field in metadata and metadata[field]:
                text_parts.append(metadata[field])
        
        return " ".join(text_parts)
    
    def semantic_search(self, query: str, top_k: int = 10, 
                       source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if not self.is_fitted or self.embeddings is None:
            logger.warning("Search index not built. No results returned.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
            
            results = []
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                
                doc = self.documents[idx]
                source = doc.get('metadata', {}).get('source', 'unknown')
                
                # Apply source filter if specified
                if source_filter and source != source_filter:
                    continue
                
                result = {
                    'document': doc,
                    'similarity_score': float(similarities[idx]),
                    'search_type': 'semantic'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 10, 
                      source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform keyword search using TF-IDF"""
        if not self.is_fitted or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not built. No results returned.")
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
            
            results = []
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                
                doc = self.documents[idx]
                source = doc.get('metadata', {}).get('source', 'unknown')
                
                # Apply source filter if specified
                if source_filter and source != source_filter:
                    continue
                
                # Skip documents with zero similarity
                if similarities[idx] == 0:
                    continue
                
                result = {
                    'document': doc,
                    'similarity_score': float(similarities[idx]),
                    'search_type': 'keyword'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     semantic_weight: float = 0.7,
                     source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        try:
            # Get results from both methods
            semantic_results = self.semantic_search(query, top_k * 2, source_filter)
            keyword_results = self.keyword_search(query, top_k * 2, source_filter)
            
            # Create a combined score dictionary
            combined_scores = {}
            
            # Add semantic scores
            for result in semantic_results:
                doc_id = self._get_document_id(result['document'])
                combined_scores[doc_id] = {
                    'document': result['document'],
                    'semantic_score': result['similarity_score'],
                    'keyword_score': 0.0
                }
            
            # Add keyword scores
            for result in keyword_results:
                doc_id = self._get_document_id(result['document'])
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword_score'] = result['similarity_score']
                else:
                    combined_scores[doc_id] = {
                        'document': result['document'],
                        'semantic_score': 0.0,
                        'keyword_score': result['similarity_score']
                    }
            
            # Calculate combined scores
            final_results = []
            for doc_id, scores in combined_scores.items():
                combined_score = (
                    semantic_weight * scores['semantic_score'] + 
                    (1 - semantic_weight) * scores['keyword_score']
                )
                
                final_results.append({
                    'document': scores['document'],
                    'similarity_score': combined_score,
                    'semantic_score': scores['semantic_score'],
                    'keyword_score': scores['keyword_score'],
                    'search_type': 'hybrid'
                })
            
            # Sort by combined score and return top k
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _get_document_id(self, document: Dict[str, Any]) -> str:
        """Generate a unique ID for a document"""
        # Use chunk_id if available, otherwise create from content hash
        if 'chunk_id' in document:
            return document['chunk_id']
        
        content = document.get('content', '')
        title = document.get('title', '')
        source = document.get('metadata', {}).get('source', '')
        
        import hashlib
        content_hash = hashlib.md5(f"{source}:{title}:{content[:100]}".encode()).hexdigest()[:12]
        return content_hash
    
    def find_similar_documents(self, document: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        if not self.is_fitted:
            return []
        
        try:
            # Extract text from the input document
            doc_text = self._extract_searchable_text(document)
            
            # Use semantic search
            results = self.semantic_search(doc_text, top_k + 1)  # +1 to exclude self
            
            # Filter out the input document itself
            doc_id = self._get_document_id(document)
            filtered_results = [
                result for result in results 
                if self._get_document_id(result['document']) != doc_id
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def search_by_category(self, category: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by category"""
        results = []
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            categories = metadata.get('categories', [])
            topics = metadata.get('topics', [])
            all_categories = categories + topics
            
            # Check if category matches (case-insensitive)
            if any(category.lower() in cat.lower() for cat in all_categories):
                results.append({
                    'document': doc,
                    'similarity_score': 1.0,  # Perfect match for category
                    'search_type': 'category'
                })
        
        return results[:top_k]
    
    def search_by_author(self, author: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by author"""
        results = []
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            authors = metadata.get('authors', [])
            
            # Check if author matches (case-insensitive)
            if any(author.lower() in auth.lower() for auth in authors):
                results.append({
                    'document': doc,
                    'similarity_score': 1.0,  # Perfect match for author
                    'search_type': 'author'
                })
        
        return results[:top_k]
    
    def advanced_search(self, query: str, filters: Dict[str, Any] = None, 
                       search_type: str = 'hybrid', top_k: int = 10) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters and options"""
        filters = filters or {}
        
        # Apply filters first
        filtered_docs = self.documents
        
        if 'source' in filters:
            filtered_docs = [
                doc for doc in filtered_docs 
                if doc.get('metadata', {}).get('source') == filters['source']
            ]
        
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            filtered_docs = [
                doc for doc in filtered_docs 
                if self._is_in_date_range(doc, start_date, end_date)
            ]
        
        if 'categories' in filters:
            filtered_categories = filters['categories']
            filtered_docs = [
                doc for doc in filtered_docs 
                if self._has_any_category(doc, filtered_categories)
            ]
        
        # Temporarily update documents for search
        original_docs = self.documents
        self.documents = filtered_docs
        self._rebuild_index()
        
        try:
            # Perform search
            if search_type == 'semantic':
                results = self.semantic_search(query, top_k)
            elif search_type == 'keyword':
                results = self.keyword_search(query, top_k)
            else:  # hybrid
                results = self.hybrid_search(query, top_k)
            
            return results
            
        finally:
            # Restore original documents
            self.documents = original_docs
            self._rebuild_index()
    
    def _is_in_date_range(self, doc: Dict[str, Any], start_date: str, end_date: str) -> bool:
        """Check if document is in date range"""
        metadata = doc.get('metadata', {})
        doc_date = metadata.get('published') or metadata.get('created_at')
        
        if not doc_date:
            return True  # Include if no date available
        
        # Simple date comparison (assumes ISO format)
        return start_date <= doc_date <= end_date
    
    def _has_any_category(self, doc: Dict[str, Any], categories: List[str]) -> bool:
        """Check if document has any of the specified categories"""
        metadata = doc.get('metadata', {})
        doc_categories = metadata.get('categories', []) + metadata.get('topics', [])
        
        return any(
            any(cat.lower() in doc_cat.lower() for doc_cat in doc_categories)
            for cat in categories
        )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        source_counts = {}
        category_counts = {}
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            
            # Count by source
            source = metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Count by categories
            categories = metadata.get('categories', []) + metadata.get('topics', [])
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'status': 'fitted',
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'source_breakdown': source_counts,
            'top_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def clear_index(self):
        """Clear the search index"""
        self.documents = []
        self.embeddings = None
        self.tfidf_matrix = None
        self.is_fitted = False
        logger.info("Search index cleared")