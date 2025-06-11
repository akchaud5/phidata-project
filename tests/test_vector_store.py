"""Tests for VectorStore class"""
import pytest
import os
import numpy
from unittest.mock import Mock, patch, MagicMock
from rag_system.vector_store import VectorStore


class TestVectorStore:
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_init(self, mock_transformer, mock_client, temp_dir):
        """Test VectorStore initialization"""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        vector_store = VectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        assert vector_store.collection_name == "test_collection"
        assert vector_store.persist_directory == temp_dir
        mock_client.assert_called_once_with(path=temp_dir)
        mock_client_instance.create_collection.assert_called_once()
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_add_documents(self, mock_transformer, mock_client, sample_documents):
        """Test adding documents to vector store"""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        mock_transformer.return_value = mock_embedding_model
        
        vector_store = VectorStore()
        ids = vector_store.add_documents(sample_documents)
        
        assert len(ids) == 2
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert len(call_args.kwargs['documents']) == 2
        assert len(call_args.kwargs['embeddings']) == 2
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_search(self, mock_transformer, mock_client):
        """Test searching documents"""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Test content']],
            'metadatas': [[{'source': 'test'}]],
            'distances': [[0.1]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = numpy.array([[0.1, 0.2]])
        mock_transformer.return_value = mock_embedding_model
        
        vector_store = VectorStore()
        results = vector_store.search("test query", n_results=1)
        
        assert len(results) == 1
        assert results[0]['id'] == 'doc1'
        assert results[0]['content'] == 'Test content'
        assert results[0]['distance'] == 0.1
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_delete_documents(self, mock_transformer, mock_client):
        """Test deleting documents"""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        vector_store = VectorStore()
        result = vector_store.delete_documents(['doc1', 'doc2'])
        
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=['doc1', 'doc2'])
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_get_collection_info(self, mock_transformer, mock_client):
        """Test getting collection information"""
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        vector_store = VectorStore(collection_name="test_collection")
        info = vector_store.get_collection_info()
        
        assert info['name'] == 'test_collection'
        assert info['count'] == 5
        assert 'persist_directory' in info
    
    @patch('rag_system.vector_store.chromadb.PersistentClient')
    @patch('rag_system.vector_store.SentenceTransformer')
    def test_clear_collection(self, mock_transformer, mock_client):
        """Test clearing collection"""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        vector_store = VectorStore(collection_name="test_collection")
        result = vector_store.clear_collection()
        
        assert result is True
        mock_client_instance.delete_collection.assert_called_once_with(name="test_collection")
        mock_client_instance.create_collection.assert_called()