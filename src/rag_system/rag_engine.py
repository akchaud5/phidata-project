import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.llm.anthropic import Claude
from phi.knowledge.base import KnowledgeBase
from phi.vectordb.chroma import ChromaDb
from .vector_store import VectorStore
from .document_processor import DocumentProcessor
from .citation_tracker import CitationTracker
from .semantic_search import SemanticSearchEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, 
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4",
                 vector_store_path: str = "./data/chroma"):
        
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.document_processor = DocumentProcessor()
        self.citation_tracker = CitationTracker(storage_path=f"{vector_store_path}/citations.json")
        self.semantic_search = SemanticSearchEngine()
        
        # Initialize LLM
        if llm_provider.lower() == "openai":
            self.llm = OpenAIChat(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif llm_provider.lower() == "anthropic":
            self.llm = Claude(
                model=model_name,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Initialize Phidata knowledge base
        self.knowledge_base = KnowledgeBase(
            vector_db=ChromaDb(
                collection="research_knowledge",
                path=vector_store_path
            )
        )
        
        # Initialize assistant
        self.assistant = Assistant(
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            description="You are an intelligent research assistant that helps users find and analyze academic papers, code repositories, and other research materials.",
            instructions=[
                "Always provide accurate and helpful information based on the retrieved context.",
                "When citing sources, include relevant details like paper titles, authors, and URLs.",
                "If you don't have enough information to answer a question, say so clearly.",
                "Provide structured responses with clear sections when appropriate.",
                "When discussing code repositories, mention key details like programming language and star count.",
                "For academic papers, summarize key findings and methodologies."
            ],
            show_tool_calls=True,
            markdown=True
        )
    
    async def add_documents_from_source(self, source_type: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents from various sources to the knowledge base"""
        try:
            all_chunks = []
            
            for doc in documents:
                if source_type == "arxiv":
                    chunks = self.document_processor.process_arxiv_paper(doc)
                elif source_type == "github":
                    chunks = self.document_processor.process_github_repo(doc)
                elif source_type == "wikipedia":
                    chunks = self.document_processor.process_wikipedia_article(doc)
                else:
                    chunks = self.document_processor.process_generic_document(
                        title=doc.get('title', 'Untitled'),
                        content=doc.get('content', ''),
                        source=source_type,
                        additional_metadata=doc.get('metadata', {})
                    )
                
                all_chunks.extend(chunks)
            
            # Add to vector store
            chunk_ids = self.vector_store.add_documents(all_chunks)
            
            # Add to semantic search index
            self.semantic_search.add_documents(all_chunks)
            
            # Add citations for original documents
            for doc in documents:
                self.citation_tracker.add_citation_from_document(
                    doc_metadata=doc,
                    response_text="",
                    relevance_score=1.0
                )
            
            # Add to Phidata knowledge base
            for chunk in all_chunks:
                await self.knowledge_base.load_text(
                    text=chunk['content'],
                    metadata=chunk
                )
            
            logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} {source_type} documents")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding documents from {source_type}: {e}")
            return []
    
    async def search_knowledge_base(self, query: str, num_results: int = 5, 
                                   source_filter: Optional[str] = None,
                                   search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant documents"""
        try:
            # Use semantic search engine for advanced search capabilities
            if search_type == "semantic":
                results = self.semantic_search.semantic_search(
                    query=query,
                    top_k=num_results,
                    source_filter=source_filter
                )
            elif search_type == "keyword":
                results = self.semantic_search.keyword_search(
                    query=query,
                    top_k=num_results,
                    source_filter=source_filter
                )
            else:  # hybrid
                results = self.semantic_search.hybrid_search(
                    query=query,
                    top_k=num_results,
                    source_filter=source_filter
                )
            
            # Convert semantic search results to vector store format
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': self.semantic_search._get_document_id(result['document']),
                    'content': result['document']['content'],
                    'metadata': result['document'].get('metadata', {}),
                    'distance': 1 - result['similarity_score'],  # Convert similarity to distance
                    'similarity_score': result['similarity_score'],
                    'search_type': result['search_type']
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            # Fallback to vector store search
            where_clause = None
            if source_filter:
                where_clause = {"source": source_filter}
            
            return self.vector_store.search(
                query=query,
                n_results=num_results,
                where=where_clause
            )
    
    async def generate_response(self, query: str, context_limit: int = 5, 
                              include_citations: bool = True) -> Dict[str, Any]:
        """Generate a response using RAG with citations"""
        try:
            # Get relevant context
            relevant_docs = await self.search_knowledge_base(query, num_results=context_limit)
            
            # Build context with citations
            context_parts = []
            citation_ids = []
            
            for i, doc in enumerate(relevant_docs):
                # Add citation for this document
                citation_id = self.citation_tracker.add_citation_from_document(
                    doc_metadata=doc['metadata'],
                    response_text=query,
                    relevance_score=doc.get('similarity_score', 0.0)
                )
                citation_ids.append(citation_id)
                
                # Format context with citation reference
                context_part = f"[{i+1}] {doc['content'][:800]}..."
                context_parts.append(context_part)
            
            # Create prompt with context
            context_text = "\n\n".join(context_parts)
            
            enhanced_query = f"""Based on the following context, please answer the user's question. Include citation numbers [1], [2], etc. when referencing specific information from the sources.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided."""
            
            # Use the Phidata assistant
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.assistant.run, enhanced_query
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Build formatted citations
            formatted_citations = []
            if include_citations and citation_ids:
                for i, citation_id in enumerate(citation_ids):
                    formatted_citation = self.citation_tracker.format_citation(citation_id, style='apa')
                    formatted_citations.append(f"[{i+1}] {formatted_citation}")
            
            return {
                'response': response_text,
                'citations': formatted_citations,
                'citation_ids': citation_ids,
                'relevant_docs': len(relevant_docs),
                'search_quality': sum(doc.get('similarity_score', 0) for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"I encountered an error while generating a response: {str(e)}",
                'citations': [],
                'citation_ids': [],
                'relevant_docs': 0,
                'search_quality': 0
            }
    
    async def generate_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response using RAG"""
        try:
            # For streaming, we'll need to implement custom logic
            # since Phidata assistant might not support streaming directly
            
            # First, get relevant context
            relevant_docs = await self.search_knowledge_base(query, num_results=3)
            
            # Build context
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('source', 'unknown')}\n"
                f"Title: {doc['metadata'].get('title', 'N/A')}\n"
                f"Content: {doc['content'][:500]}..."
                for doc in relevant_docs
            ])
            
            # Create prompt with context
            prompt = f"""Based on the following context, please answer the user's question:

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. Include relevant citations and source information."""

            # Generate response (this would need to be adapted for actual streaming)
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.assistant.run, prompt
            )
            
            # Yield response in chunks (simulated streaming)
            content = response.content if hasattr(response, 'content') else str(response)
            for i in range(0, len(content), 50):
                yield content[i:i+50]
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            vector_info = self.vector_store.get_collection_info()
            semantic_stats = self.semantic_search.get_search_statistics()
            citation_stats = self.citation_tracker.get_citation_statistics()
            
            # Get source breakdown
            all_docs = self.vector_store.search("", n_results=1000)  # Get all docs
            source_counts = {}
            for doc in all_docs:
                source = doc['metadata'].get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                'total_documents': vector_info.get('count', 0),
                'collection_name': vector_info.get('name', ''),
                'source_breakdown': source_counts,
                'persist_directory': vector_info.get('persist_directory', ''),
                'semantic_search': semantic_stats,
                'citations': citation_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {}
    
    async def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base"""
        try:
            # Clear vector store
            vector_success = self.vector_store.clear_collection()
            
            # Clear semantic search index
            self.semantic_search.clear_index()
            
            # Clear citations
            self.citation_tracker.clear_citations()
            
            if vector_success:
                logger.info("Knowledge base cleared successfully")
            return vector_success
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
    
    # New methods for enhanced functionality
    
    async def search_by_similarity(self, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            # Find the document in semantic search index
            target_doc = None
            for doc in self.semantic_search.documents:
                if self.semantic_search._get_document_id(doc) == document_id:
                    target_doc = doc
                    break
            
            if not target_doc:
                logger.warning(f"Document {document_id} not found")
                return []
            
            results = self.semantic_search.find_similar_documents(target_doc, top_k)
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    async def search_by_category(self, category: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by category"""
        try:
            results = self.semantic_search.search_by_category(category, top_k)
            return results
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return []
    
    async def search_by_author(self, author: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by author"""
        try:
            results = self.semantic_search.search_by_author(author, top_k)
            return results
        except Exception as e:
            logger.error(f"Error searching by author: {e}")
            return []
    
    def get_citation_bibliography(self, citation_ids: List[str], style: str = 'apa') -> str:
        """Generate a bibliography for given citation IDs"""
        return self.citation_tracker.export_bibliography(citation_ids, style)
    
    def search_citations(self, query: str) -> List[Dict[str, Any]]:
        """Search citations by content"""
        from dataclasses import asdict
        citations = self.citation_tracker.search_citations(query)
        return [asdict(citation) for citation in citations]

# Import asyncio at the top of the file
import asyncio