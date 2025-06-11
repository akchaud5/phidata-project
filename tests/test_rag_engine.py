"""Tests for RAGEngine class"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from rag_system.rag_engine import RAGEngine


class TestRAGEngine:
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_init_openai(self, mock_assistant, mock_openai, mock_semantic, 
                        mock_citation, mock_processor, mock_vector):
        """Test RAGEngine initialization with OpenAI"""
        engine = RAGEngine(llm_provider="openai", model_name="gpt-4")
        
        assert engine.vector_store is not None
        assert engine.document_processor is not None
        assert engine.citation_tracker is not None
        assert engine.semantic_search is not None
        mock_openai.assert_called_once()
        mock_assistant.assert_called_once()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.Claude')
    @patch('rag_system.rag_engine.Assistant')
    def test_init_anthropic(self, mock_assistant, mock_claude, mock_semantic,
                           mock_citation, mock_processor, mock_vector):
        """Test RAGEngine initialization with Anthropic"""
        engine = RAGEngine(llm_provider="anthropic", model_name="claude-3")
        
        mock_claude.assert_called_once()
        mock_assistant.assert_called_once()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    def test_init_invalid_provider(self, mock_semantic, mock_citation, mock_processor, mock_vector):
        """Test RAGEngine initialization with invalid provider"""
        with pytest.raises(ValueError):
            RAGEngine(llm_provider="invalid_provider")
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_add_documents_from_source_arxiv(self, mock_assistant, mock_openai,
                                                  mock_semantic, mock_citation,
                                                  mock_processor, mock_vector,
                                                  sample_arxiv_papers):
        """Test adding arXiv documents"""
        # Setup mocks
        mock_processor_instance = mock_processor.return_value
        mock_processor_instance.process_arxiv_paper.return_value = [
            {'id': 'chunk1', 'content': 'chunk content', 'metadata': {}}
        ]
        
        mock_vector_instance = mock_vector.return_value
        mock_vector_instance.add_documents.return_value = ['chunk1']
        
        mock_semantic_instance = mock_semantic.return_value
        mock_citation_instance = mock_citation.return_value
        
        engine = RAGEngine()
        
        result = await engine.add_documents_from_source("arxiv", sample_arxiv_papers)
        
        assert result == ['chunk1']
        mock_processor_instance.process_arxiv_paper.assert_called()
        mock_vector_instance.add_documents.assert_called()
        mock_semantic_instance.add_documents.assert_called()
        mock_citation_instance.add_citation_from_document.assert_called()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_search_knowledge_base_semantic(self, mock_assistant, mock_openai,
                                                 mock_semantic, mock_citation,
                                                 mock_processor, mock_vector):
        """Test semantic search in knowledge base"""
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.semantic_search.return_value = [
            {
                'document': {'content': 'test content', 'metadata': {}},
                'similarity_score': 0.9,
                'search_type': 'semantic'
            }
        ]
        mock_semantic_instance._get_document_id.return_value = 'doc1'
        
        engine = RAGEngine()
        
        results = await engine.search_knowledge_base(
            "test query", 
            num_results=5, 
            search_type="semantic"
        )
        
        assert len(results) == 1
        assert results[0]['similarity_score'] == 0.9
        assert results[0]['search_type'] == 'semantic'
        mock_semantic_instance.semantic_search.assert_called_once()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_assistant, mock_openai,
                                   mock_semantic, mock_citation,
                                   mock_processor, mock_vector):
        """Test generating response with RAG"""
        # Setup mocks
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.hybrid_search.return_value = [
            {
                'document': {
                    'content': 'relevant document content',
                    'metadata': {'title': 'Test Doc', 'source': 'test'}
                },
                'similarity_score': 0.8,
                'search_type': 'hybrid'
            }
        ]
        mock_semantic_instance._get_document_id.return_value = 'doc1'
        
        mock_citation_instance = mock_citation.return_value
        mock_citation_instance.add_citation_from_document.return_value = 'citation1'
        mock_citation_instance.format_citation.return_value = 'Test Citation Format'
        
        mock_assistant_instance = mock_assistant.return_value
        mock_response = Mock()
        mock_response.content = "This is the AI response with citations [1]."
        mock_assistant_instance.run.return_value = mock_response
        
        engine = RAGEngine()
        
        # Mock asyncio.get_event_loop().run_in_executor to return the mock response directly
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=mock_response)
            mock_loop.return_value.run_in_executor = mock_executor
            
            result = await engine.generate_response("test query")
        
        assert 'response' in result
        assert 'citations' in result
        assert len(result['citations']) > 0
        assert result['relevant_docs'] == 1
        mock_assistant_instance.run.assert_called()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_get_knowledge_base_stats(self, mock_assistant, mock_openai,
                                     mock_semantic, mock_citation,
                                     mock_processor, mock_vector):
        """Test getting knowledge base statistics"""
        mock_vector_instance = mock_vector.return_value
        mock_vector_instance.get_collection_info.return_value = {
            'count': 100,
            'name': 'test_collection',
            'persist_directory': './test_data'
        }
        mock_vector_instance.search.return_value = [
            {'metadata': {'source': 'arxiv'}},
            {'metadata': {'source': 'github'}},
            {'metadata': {'source': 'arxiv'}}
        ]
        
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.get_search_statistics.return_value = {'total_docs': 100}
        
        mock_citation_instance = mock_citation.return_value
        mock_citation_instance.get_citation_statistics.return_value = {'total_citations': 50}
        
        engine = RAGEngine()
        
        stats = engine.get_knowledge_base_stats()
        
        assert stats['total_documents'] == 100
        assert stats['collection_name'] == 'test_collection'
        assert 'source_breakdown' in stats
        assert stats['source_breakdown']['arxiv'] == 2
        assert stats['source_breakdown']['github'] == 1
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_clear_knowledge_base(self, mock_assistant, mock_openai,
                                       mock_semantic, mock_citation,
                                       mock_processor, mock_vector):
        """Test clearing knowledge base"""
        mock_vector_instance = mock_vector.return_value
        mock_vector_instance.clear_collection.return_value = True
        
        mock_semantic_instance = mock_semantic.return_value
        mock_citation_instance = mock_citation.return_value
        
        engine = RAGEngine()
        
        result = await engine.clear_knowledge_base()
        
        assert result is True
        mock_vector_instance.clear_collection.assert_called_once()
        mock_semantic_instance.clear_index.assert_called_once()
        mock_citation_instance.clear_citations.assert_called_once()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_search_by_similarity(self, mock_assistant, mock_openai,
                                       mock_semantic, mock_citation,
                                       mock_processor, mock_vector):
        """Test finding similar documents"""
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.documents = [
            {'id': 'target_doc', 'content': 'target content'}
        ]
        mock_semantic_instance._get_document_id.return_value = 'target_doc'
        mock_semantic_instance.find_similar_documents.return_value = [
            {'similarity_score': 0.9, 'document': {'content': 'similar content'}}
        ]
        
        engine = RAGEngine()
        
        results = await engine.search_by_similarity('target_doc', top_k=5)
        
        assert len(results) == 1
        assert results[0]['similarity_score'] == 0.9
        mock_semantic_instance.find_similar_documents.assert_called_once()
    
    @patch('rag_system.rag_engine.VectorStore')
    @patch('rag_system.rag_engine.DocumentProcessor')
    @patch('rag_system.rag_engine.CitationTracker')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_get_citation_bibliography(self, mock_assistant, mock_openai,
                                      mock_semantic, mock_citation,
                                      mock_processor, mock_vector):
        """Test generating bibliography"""
        mock_citation_instance = mock_citation.return_value
        mock_citation_instance.export_bibliography.return_value = "Bibliography content"
        
        engine = RAGEngine()
        
        bibliography = engine.get_citation_bibliography(['citation1', 'citation2'], 'apa')
        
        assert bibliography == "Bibliography content"
        mock_citation_instance.export_bibliography.assert_called_once_with(['citation1', 'citation2'], 'apa')