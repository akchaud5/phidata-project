"""Test configuration and fixtures"""
import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock = Mock()
    mock.add_documents.return_value = ['doc1', 'doc2', 'doc3']
    mock.search.return_value = [
        {
            'id': 'doc1',
            'content': 'Test document content',
            'metadata': {'source': 'test', 'title': 'Test Doc'},
            'distance': 0.1
        }
    ]
    mock.get_collection_info.return_value = {
        'name': 'test_collection',
        'count': 10,
        'persist_directory': './test_data'
    }
    mock.clear_collection.return_value = True
    return mock

@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model"""
    mock = Mock()
    mock.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'id': 'doc1',
            'content': 'This is a test document about machine learning',
            'title': 'ML Test Paper',
            'source': 'arxiv',
            'authors': 'Test Author',
            'url': 'https://arxiv.org/abs/test'
        },
        {
            'id': 'doc2',
            'content': 'This is another test document about neural networks',
            'title': 'NN Test Paper',
            'source': 'arxiv',
            'authors': 'Another Author',
            'url': 'https://arxiv.org/abs/test2'
        }
    ]

@pytest.fixture
def sample_arxiv_papers():
    """Sample arXiv papers for testing"""
    return [
        {
            'id': 'arxiv:1234.5678',
            'title': 'Attention Is All You Need',
            'summary': 'We propose a new network architecture, the Transformer...',
            'authors': 'Ashish Vaswani, Noam Shazeer',
            'published': '2017-06-12',
            'url': 'https://arxiv.org/abs/1706.03762',
            'categories': 'cs.CL',
            'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf'
        }
    ]

@pytest.fixture
def sample_github_repos():
    """Sample GitHub repositories for testing"""
    return [
        {
            'id': 'repo123',
            'name': 'transformers',
            'full_name': 'huggingface/transformers',
            'description': 'Transformers: State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.',
            'html_url': 'https://github.com/huggingface/transformers',
            'stargazers_count': 50000,
            'language': 'Python',
            'topics': ['machine-learning', 'pytorch', 'tensorflow']
        }
    ]

@pytest.fixture
def sample_wikipedia_articles():
    """Sample Wikipedia articles for testing"""
    return [
        {
            'title': 'Machine Learning',
            'summary': 'Machine learning is a method of data analysis...',
            'url': 'https://en.wikipedia.org/wiki/Machine_learning',
            'content': 'Machine learning (ML) is a field of inquiry devoted to understanding...'
        }
    ]

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock_response = Mock()
    mock_response.content = "This is a test response from the AI assistant."
    mock.run.return_value = mock_response
    return mock

@pytest.fixture
def mock_assistant():
    """Mock Phidata assistant"""
    mock = Mock()
    mock_response = Mock()
    mock_response.content = "This is a test response with proper citations [1]."
    mock.run.return_value = mock_response
    return mock