import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-\'\"]+', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        text = self.clean_text(text)
        
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max size, save current chunk
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk_data = {
                    'content': current_chunk.strip(),
                    'token_count': current_tokens,
                    'chunk_id': self._generate_chunk_id(current_chunk),
                    'created_at': datetime.now().isoformat()
                }
                
                if metadata:
                    chunk_data.update(metadata)
                
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'chunk_id': self._generate_chunk_id(current_chunk),
                'created_at': datetime.now().isoformat()
            }
            
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
        
        return chunks
    
    def _generate_chunk_id(self, text: str) -> str:
        """Generate a unique ID for a text chunk"""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = min(self.chunk_overlap // 5, len(words))  # Rough estimation
        return " ".join(words[-overlap_words:])
    
    def process_arxiv_paper(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process arXiv paper into chunks"""
        content = f"Title: {paper_data.get('title', '')}\n\n"
        content += f"Authors: {', '.join(paper_data.get('authors', []))}\n\n"
        content += f"Abstract: {paper_data.get('summary', '')}"
        
        metadata = {
            'source': 'arxiv',
            'paper_id': paper_data.get('id', ''),
            'title': paper_data.get('title', ''),
            'authors': paper_data.get('authors', []),
            'categories': paper_data.get('categories', []),
            'published': paper_data.get('published', ''),
            'pdf_url': paper_data.get('pdf_url', ''),
            'doi': paper_data.get('doi', '')
        }
        
        return self.chunk_text(content, metadata)
    
    def process_github_repo(self, repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process GitHub repository into chunks"""
        content = f"Repository: {repo_data.get('full_name', '')}\n\n"
        content += f"Description: {repo_data.get('description', '')}\n\n"
        content += f"Language: {repo_data.get('language', '')}\n"
        content += f"Stars: {repo_data.get('stars', 0)}\n"
        content += f"Topics: {', '.join(repo_data.get('topics', []))}"
        
        metadata = {
            'source': 'github',
            'repo_id': repo_data.get('id', ''),
            'full_name': repo_data.get('full_name', ''),
            'language': repo_data.get('language', ''),
            'stars': repo_data.get('stars', 0),
            'html_url': repo_data.get('html_url', ''),
            'topics': repo_data.get('topics', [])
        }
        
        return self.chunk_text(content, metadata)
    
    def process_wikipedia_article(self, article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Wikipedia article into chunks"""
        content = f"Title: {article_data.get('title', '')}\n\n"
        content += article_data.get('content', article_data.get('summary', ''))
        
        metadata = {
            'source': 'wikipedia',
            'title': article_data.get('title', ''),
            'url': article_data.get('url', ''),
            'categories': article_data.get('categories', [])
        }
        
        return self.chunk_text(content, metadata)
    
    def process_generic_document(self, title: str, content: str, source: str = 'generic', 
                                additional_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Process a generic document into chunks"""
        full_content = f"Title: {title}\n\n{content}"
        
        metadata = {
            'source': source,
            'title': title
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return self.chunk_text(full_content, metadata)