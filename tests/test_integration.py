"""Integration tests for the research assistant system"""
import pytest
import tempfile
import shutil
import asyncio
import numpy
from unittest.mock import Mock, patch, AsyncMock
from rag_system.rag_engine import RAGEngine
from rag_system.mcp_integration import MCPIntegration


class TestSystemIntegration:
    
    @pytest.fixture
    def temp_system_dir(self):
        """Create a temporary directory for system testing"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_rag_engine_initialization(self, mock_assistant, mock_openai, 
                                      mock_transformer, mock_client, temp_system_dir):
        """Test full RAG engine initialization"""
        # Setup mocks
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        
        assert engine.vector_store is not None
        assert engine.document_processor is not None
        assert engine.citation_tracker is not None
        assert engine.semantic_search is not None
        assert engine.assistant is not None
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self, mock_assistant, mock_openai,
                                                  mock_transformer, mock_client,
                                                  temp_system_dir, sample_documents):
        """Test complete document processing workflow"""
        # Setup mocks
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        mock_transformer.return_value = mock_embedding_model
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        
        # Add documents
        result = await engine.add_documents_from_source("test", sample_documents)
        
        assert len(result) > 0
        mock_collection.add.assert_called()
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @patch('rag_system.rag_engine.CitationTracker')
    @pytest.mark.asyncio
    async def test_search_and_response_generation(self, mock_citation, mock_semantic, mock_assistant, mock_openai,
                                                  mock_transformer, mock_client,
                                                  temp_system_dir):
        """Test search and response generation workflow"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Test document content about machine learning']],
            'metadatas': [[{'source': 'test', 'title': 'Test Document'}]],
            'distances': [[0.1]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = numpy.array([[0.1, 0.2]])
        mock_transformer.return_value = mock_embedding_model
        
        # Mock semantic search
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.hybrid_search.return_value = [
            {
                'document': {
                    'content': 'Test document content about machine learning',
                    'metadata': {'source': 'test', 'title': 'Test Document'}
                },
                'similarity_score': 0.9,
                'search_type': 'hybrid'
            }
        ]
        mock_semantic_instance._get_document_id.return_value = 'doc1'
        
        # Mock citation tracker
        mock_citation_instance = mock_citation.return_value
        mock_citation_instance.add_citation_from_document.return_value = 'citation1'
        mock_citation_instance.format_citation.return_value = 'Test Citation Format'
        
        mock_assistant_instance = mock_assistant.return_value
        mock_response = Mock()
        mock_response.content = "Machine learning is a subset of artificial intelligence [1]."
        mock_assistant_instance.run.return_value = mock_response
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        
        # Mock asyncio executor
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=mock_response)
            mock_loop.return_value.run_in_executor = mock_executor
            
            response = await engine.generate_response("What is machine learning?")
        
        assert 'response' in response
        assert 'citations' in response
        assert response['relevant_docs'] > 0
        # Verify semantic search was called instead of direct vector store
        mock_semantic_instance.hybrid_search.assert_called()
        mock_executor.assert_called()
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_mcp_integration_with_rag_engine(self, mock_assistant, mock_openai,
                                            mock_transformer, mock_client,
                                            temp_system_dir):
        """Test MCP integration with RAG engine"""
        # Setup mocks
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        mcp_integration = MCPIntegration(engine)
        
        assert mcp_integration.rag_engine == engine
        assert mcp_integration.conversation_memory is not None
        assert mcp_integration.active_servers == {}
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @pytest.mark.asyncio
    async def test_conversation_session_workflow(self, mock_assistant, mock_openai,
                                                 mock_transformer, mock_client,
                                                 temp_system_dir):
        """Test complete conversation session workflow"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Relevant content']],
            'metadatas': [[{'source': 'test'}]],
            'distances': [[0.2]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_embedding_model
        
        mock_assistant_instance = mock_assistant.return_value
        mock_response = Mock()
        mock_response.content = "Here's your answer based on the context."
        mock_assistant_instance.run.return_value = mock_response
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        mcp_integration = MCPIntegration(engine)
        
        # Create session
        session_id = mcp_integration.create_conversation_session(
            user_id="test_user",
            title="Test Session"
        )
        assert session_id is not None
        
        # Mock multi-source search
        mcp_integration.multi_source_search = AsyncMock(return_value={})
        
        # Mock asyncio executor
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=mock_response)
            mock_loop.return_value.run_in_executor = mock_executor
            
            # Enhanced query
            response = await mcp_integration.enhanced_query(
                "Test question",
                auto_search=False,
                session_id=session_id
            )
        
        assert response['session_id'] == session_id
        assert 'response' in response
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    def test_knowledge_base_statistics(self, mock_assistant, mock_openai,
                                      mock_transformer, mock_client,
                                      temp_system_dir):
        """Test knowledge base statistics collection"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['content1', 'content2']],
            'metadatas': [[
                {'source': 'arxiv', 'title': 'Paper 1'},
                {'source': 'github', 'title': 'Repo 1'}
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_embedding_model
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        
        stats = engine.get_knowledge_base_stats()
        
        assert stats['total_documents'] == 42
        assert 'source_breakdown' in stats
        assert 'semantic_search' in stats
        assert 'citations' in stats
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    @patch('rag_system.rag_engine.OpenAIChat')
    @patch('rag_system.rag_engine.Assistant')
    @patch('rag_system.rag_engine.SemanticSearchEngine')
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_semantic, mock_assistant, mock_openai,
                                          mock_transformer, mock_client,
                                          temp_system_dir):
        """Test error handling in the complete workflow"""
        # Setup mocks to simulate errors
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Vector store error")
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Mock semantic search to also fail
        mock_semantic_instance = mock_semantic.return_value
        mock_semantic_instance.hybrid_search.side_effect = Exception("Semantic search error")
        
        engine = RAGEngine(vector_store_path=temp_system_dir)
        
        # Test search error handling
        results = await engine.search_knowledge_base("test query")
        assert results == []  # Should return empty list on error
        
        # Test response generation error handling
        response = await engine.generate_response("test query")
        # Since search fails but doesn't propagate up, we should still get a response
        # but with no relevant documents found
        assert response['relevant_docs'] == 0
        assert 'response' in response
    
    @pytest.mark.asyncio
    async def test_system_performance_baseline(self, temp_system_dir):
        """Test basic system performance metrics"""
        import time
        
        with patch('rag_system.vector_store.chromadb.PersistentClient'), \
             patch('rag_system.vector_store.SentenceTransformer'), \
             patch('rag_system.rag_engine.OpenAIChat'), \
             patch('rag_system.rag_engine.Assistant'):
            
            start_time = time.time()
            engine = RAGEngine(vector_store_path=temp_system_dir)
            init_time = time.time() - start_time
            
            # Initialization should be reasonably fast
            assert init_time < 5.0  # 5 seconds max for initialization
            
            mcp_integration = MCPIntegration(engine)
            
            # Session creation should be fast
            start_time = time.time()
            session_id = mcp_integration.create_conversation_session("test_user")
            session_time = time.time() - start_time
            
            assert session_time < 1.0  # 1 second max for session creation
            assert session_id is not None