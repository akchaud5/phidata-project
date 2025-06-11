"""Tests for CitationTracker class"""
import pytest
import os
import json
from unittest.mock import Mock, patch
from rag_system.citation_tracker import CitationTracker, Citation


class TestCitationTracker:
    
    def test_init(self, temp_dir):
        """Test CitationTracker initialization"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        assert tracker.storage_path == storage_path
        assert tracker.citations == {}
    
    def test_add_citation_from_document(self, temp_dir):
        """Test adding citation from document metadata"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        doc_metadata = {
            'title': 'Test Paper',
            'authors': 'John Doe, Jane Smith',
            'source': 'arxiv',
            'url': 'https://arxiv.org/abs/test',
            'published': '2023-01-01'
        }
        
        citation_id = tracker.add_citation_from_document(
            doc_metadata=doc_metadata,
            response_text="This paper discusses...",
            relevance_score=0.9
        )
        
        assert citation_id is not None
        assert len(tracker.citations) == 1
        
        citation = tracker.citations[citation_id]
        assert citation.title == 'Test Paper'
        assert citation.authors == ['John Doe, Jane Smith']
        assert citation.source == 'arxiv'
        assert citation.relevance_score == 0.9
    
    def test_format_citation_apa(self, temp_dir):
        """Test APA citation formatting"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        doc_metadata = {
            'title': 'Test Paper',
            'authors': 'Doe, J., & Smith, J.',
            'source': 'arxiv',
            'url': 'https://arxiv.org/abs/test',
            'published': '2023'
        }
        
        citation_id = tracker.add_citation_from_document(doc_metadata, "", 1.0)
        formatted = tracker.format_citation(citation_id, style='apa')
        
        assert 'Doe, J., & Smith, J.' in formatted
        assert '(2023)' in formatted
        assert 'Test Paper' in formatted
    
    def test_format_citation_mla(self, temp_dir):
        """Test MLA citation formatting"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        doc_metadata = {
            'title': 'Test Paper',
            'authors': 'John Doe, Jane Smith',
            'source': 'arxiv',
            'url': 'https://arxiv.org/abs/test',
            'published': '2023'
        }
        
        citation_id = tracker.add_citation_from_document(doc_metadata, "", 1.0)
        formatted = tracker.format_citation(citation_id, style='mla')
        
        assert 'John Doe, Jane Smith' in formatted
        assert 'Test Paper' in formatted
        assert '2023' in formatted
    
    def test_export_bibliography(self, temp_dir):
        """Test bibliography export"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        # Add multiple citations
        citation_ids = []
        for i in range(3):
            doc_metadata = {
                'title': f'Test Paper {i+1}',
                'authors': f'Author {i+1}',
                'source': 'arxiv',
                'url': f'https://arxiv.org/abs/test{i+1}',
                'published': '2023'
            }
            citation_id = tracker.add_citation_from_document(doc_metadata, "", 1.0)
            citation_ids.append(citation_id)
        
        bibliography = tracker.export_bibliography(citation_ids, style='apa')
        
        assert 'Test Paper 1' in bibliography
        assert 'Test Paper 2' in bibliography
        assert 'Test Paper 3' in bibliography
        assert 'Author 1' in bibliography
    
    def test_search_citations(self, temp_dir):
        """Test searching citations"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        # Add citations with different content
        doc1 = {
            'title': 'Machine Learning Paper',
            'authors': 'ML Author',
            'source': 'arxiv'
        }
        doc2 = {
            'title': 'Deep Learning Study',
            'authors': 'DL Author',
            'source': 'arxiv'
        }
        
        tracker.add_citation_from_document(doc1, "machine learning content", 1.0)
        tracker.add_citation_from_document(doc2, "deep learning content", 1.0)
        
        # Search for machine learning
        results = tracker.search_citations("machine learning")
        assert len(results) >= 1
        assert any("Machine Learning" in citation.title for citation in results)
    
    def test_get_citation_statistics(self, temp_dir):
        """Test getting citation statistics"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        # Add citations from different sources
        sources = ['arxiv', 'github', 'wikipedia']
        for source in sources:
            doc_metadata = {
                'title': f'Test {source} document',
                'authors': 'Test Author',
                'source': source
            }
            tracker.add_citation_from_document(doc_metadata, "", 0.8)
        
        stats = tracker.get_citation_statistics()
        
        assert stats['total_citations'] == 3
        assert 'arxiv' in stats['source_breakdown']
        assert 'github' in stats['source_breakdown']
        assert 'wikipedia' in stats['source_breakdown']
        assert stats['average_relevance'] == 0.8
    
    def test_save_and_load_citations(self, temp_dir):
        """Test saving and loading citations from file"""
        storage_path = os.path.join(temp_dir, "citations.json")
        
        # Create tracker and add citation
        tracker1 = CitationTracker(storage_path=storage_path)
        doc_metadata = {
            'title': 'Persistent Test Paper',
            'authors': 'Test Author',
            'source': 'arxiv'
        }
        citation_id = tracker1.add_citation_from_document(doc_metadata, "", 1.0)
        
        # Save citations
        tracker1._save_citations()
        
        # Create new tracker and load citations
        tracker2 = CitationTracker(storage_path=storage_path)
        
        assert len(tracker2.citations) == 1
        assert tracker2.citations[0].title == 'Persistent Test Paper'
        assert tracker2.citations[0].authors == 'Test Author'
    
    def test_clear_citations(self, temp_dir):
        """Test clearing all citations"""
        storage_path = os.path.join(temp_dir, "citations.json")
        tracker = CitationTracker(storage_path=storage_path)
        
        # Add some citations
        for i in range(3):
            doc_metadata = {
                'title': f'Test Paper {i+1}',
                'authors': f'Author {i+1}',
                'source': 'arxiv'
            }
            tracker.add_citation_from_document(doc_metadata, "", 1.0)
        
        assert len(tracker.citations) == 3
        
        # Clear citations
        tracker.clear_citations()
        
        assert len(tracker.citations) == 0