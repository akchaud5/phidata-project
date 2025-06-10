import json
import hashlib
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Citation:
    id: str
    source: str
    title: str
    authors: List[str]
    url: str
    publication_date: Optional[str]
    doi: Optional[str]
    abstract: Optional[str]
    categories: List[str]
    cited_in_response: str
    relevance_score: float
    created_at: str

class CitationTracker:
    def __init__(self, storage_path: str = "./data/citations.json"):
        self.storage_path = storage_path
        self.citations: Dict[str, Citation] = {}
        self.load_citations()
    
    def load_citations(self):
        """Load citations from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for citation_data in data:
                    citation = Citation(**citation_data)
                    self.citations[citation.id] = citation
            logger.info(f"Loaded {len(self.citations)} citations from storage")
        except FileNotFoundError:
            logger.info("No existing citations file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading citations: {e}")
    
    def save_citations(self):
        """Save citations to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                citations_list = [asdict(citation) for citation in self.citations.values()]
                json.dump(citations_list, f, indent=2)
            logger.info(f"Saved {len(self.citations)} citations to storage")
        except Exception as e:
            logger.error(f"Error saving citations: {e}")
    
    def generate_citation_id(self, title: str, source: str) -> str:
        """Generate a unique citation ID"""
        content = f"{source}:{title}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_citation_from_document(self, doc_metadata: Dict[str, Any], 
                                 response_text: str, relevance_score: float = 0.0) -> str:
        """Add citation from document metadata"""
        try:
            source = doc_metadata.get('source', 'unknown')
            title = doc_metadata.get('title', 'Untitled')
            
            citation_id = self.generate_citation_id(title, source)
            
            # Extract relevant information based on source
            if source == 'arxiv':
                authors = doc_metadata.get('authors', [])
                url = doc_metadata.get('pdf_url', '')
                publication_date = doc_metadata.get('published', '')
                doi = doc_metadata.get('doi', '')
                abstract = doc_metadata.get('summary', '')
                categories = doc_metadata.get('categories', [])
            elif source == 'github':
                authors = [doc_metadata.get('full_name', '').split('/')[0]] if doc_metadata.get('full_name') else []
                url = doc_metadata.get('html_url', '')
                publication_date = doc_metadata.get('created_at', '')
                doi = None
                abstract = doc_metadata.get('description', '')
                categories = doc_metadata.get('topics', [])
            elif source == 'wikipedia':
                authors = ['Wikipedia Contributors']
                url = doc_metadata.get('url', '')
                publication_date = None
                doi = None
                abstract = doc_metadata.get('summary', '')
                categories = doc_metadata.get('categories', [])
            else:
                authors = []
                url = ''
                publication_date = None
                doi = None
                abstract = ''
                categories = []
            
            citation = Citation(
                id=citation_id,
                source=source,
                title=title,
                authors=authors,
                url=url,
                publication_date=publication_date,
                doi=doi,
                abstract=abstract,
                categories=categories,
                cited_in_response=response_text[:500] + "..." if len(response_text) > 500 else response_text,
                relevance_score=relevance_score,
                created_at=datetime.now().isoformat()
            )
            
            self.citations[citation_id] = citation
            self.save_citations()
            
            return citation_id
            
        except Exception as e:
            logger.error(f"Error adding citation: {e}")
            return ""
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get citation by ID"""
        return self.citations.get(citation_id)
    
    def format_citation(self, citation_id: str, style: str = 'apa') -> str:
        """Format citation in specified style"""
        citation = self.get_citation(citation_id)
        if not citation:
            return f"[Citation {citation_id} not found]"
        
        try:
            if style.lower() == 'apa':
                return self._format_apa(citation)
            elif style.lower() == 'mla':
                return self._format_mla(citation)
            elif style.lower() == 'chicago':
                return self._format_chicago(citation)
            else:
                return self._format_simple(citation)
        except Exception as e:
            logger.error(f"Error formatting citation {citation_id}: {e}")
            return f"[Error formatting citation {citation_id}]"
    
    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style"""
        if citation.source == 'arxiv':
            authors_str = self._format_authors_apa(citation.authors)
            year = self._extract_year(citation.publication_date)
            return f"{authors_str} ({year}). {citation.title}. arXiv preprint. {citation.url}"
        
        elif citation.source == 'github':
            authors_str = self._format_authors_apa(citation.authors)
            year = self._extract_year(citation.publication_date)
            return f"{authors_str} ({year}). {citation.title} [Computer software]. GitHub. {citation.url}"
        
        elif citation.source == 'wikipedia':
            year = datetime.now().year  # Use current year for Wikipedia
            return f"Wikipedia contributors. ({year}). {citation.title}. In Wikipedia, The Free Encyclopedia. {citation.url}"
        
        else:
            authors_str = self._format_authors_apa(citation.authors)
            return f"{authors_str} {citation.title}. {citation.url}"
    
    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style"""
        if citation.source == 'arxiv':
            authors_str = self._format_authors_mla(citation.authors)
            return f'{authors_str} "{citation.title}." arXiv preprint, {self._extract_year(citation.publication_date)}. Web. {citation.url}'
        
        elif citation.source == 'github':
            authors_str = self._format_authors_mla(citation.authors)
            return f'{authors_str} "{citation.title}." GitHub, {self._extract_year(citation.publication_date)}. Web. {citation.url}'
        
        elif citation.source == 'wikipedia':
            return f'"{citation.title}." Wikipedia, Wikimedia Foundation, {datetime.now().strftime("%d %b %Y")}. Web. {citation.url}'
        
        else:
            authors_str = self._format_authors_mla(citation.authors)
            return f'{authors_str} "{citation.title}." Web. {citation.url}'
    
    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style"""
        if citation.source == 'arxiv':
            authors_str = self._format_authors_chicago(citation.authors)
            return f'{authors_str} "{citation.title}." arXiv preprint ({self._extract_year(citation.publication_date)}). {citation.url}'
        
        elif citation.source == 'github':
            authors_str = self._format_authors_chicago(citation.authors)
            return f'{authors_str} "{citation.title}." GitHub repository ({self._extract_year(citation.publication_date)}). {citation.url}'
        
        elif citation.source == 'wikipedia':
            return f'Wikipedia contributors. "{citation.title}." Wikipedia, The Free Encyclopedia. Accessed {datetime.now().strftime("%B %d, %Y")}. {citation.url}'
        
        else:
            authors_str = self._format_authors_chicago(citation.authors)
            return f'{authors_str} "{citation.title}." {citation.url}'
    
    def _format_simple(self, citation: Citation) -> str:
        """Format citation in simple style"""
        authors_str = ", ".join(citation.authors) if citation.authors else "Unknown"
        return f"{authors_str}. \"{citation.title}\". {citation.source.title()}. {citation.url}"
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors for APA style"""
        if not authors:
            return "Unknown"
        if len(authors) == 1:
            return self._lastname_first(authors[0])
        elif len(authors) <= 20:
            formatted = [self._lastname_first(authors[0])]
            formatted.extend([self._firstname_first(author) for author in authors[1:]])
            return ", ".join(formatted)
        else:
            return f"{self._lastname_first(authors[0])}, et al."
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors for MLA style"""
        if not authors:
            return "Unknown"
        if len(authors) == 1:
            return self._lastname_first(authors[0])
        else:
            return f"{self._lastname_first(authors[0])}, et al."
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Format authors for Chicago style"""
        if not authors:
            return "Unknown"
        if len(authors) == 1:
            return self._lastname_first(authors[0])
        elif len(authors) <= 3:
            formatted = [self._lastname_first(authors[0])]
            formatted.extend([self._firstname_first(author) for author in authors[1:]])
            return ", ".join(formatted)
        else:
            return f"{self._lastname_first(authors[0])}, et al."
    
    def _lastname_first(self, name: str) -> str:
        """Format name as 'Last, First'"""
        parts = name.strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return name
    
    def _firstname_first(self, name: str) -> str:
        """Format name as 'First Last'"""
        return name.strip()
    
    def _extract_year(self, date_str: Optional[str]) -> str:
        """Extract year from date string"""
        if not date_str:
            return "n.d."
        
        # Try to extract year using regex
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return year_match.group()
        
        return "n.d."
    
    def get_citations_by_source(self, source: str) -> List[Citation]:
        """Get all citations from a specific source"""
        return [citation for citation in self.citations.values() if citation.source == source]
    
    def search_citations(self, query: str) -> List[Citation]:
        """Search citations by title, authors, or abstract"""
        query_lower = query.lower()
        results = []
        
        for citation in self.citations.values():
            # Search in title
            if query_lower in citation.title.lower():
                results.append(citation)
                continue
            
            # Search in authors
            if any(query_lower in author.lower() for author in citation.authors):
                results.append(citation)
                continue
            
            # Search in abstract
            if citation.abstract and query_lower in citation.abstract.lower():
                results.append(citation)
                continue
        
        # Sort by relevance score
        results.sort(key=lambda c: c.relevance_score, reverse=True)
        return results
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about citations"""
        total_citations = len(self.citations)
        source_counts = {}
        category_counts = {}
        
        for citation in self.citations.values():
            # Count by source
            source_counts[citation.source] = source_counts.get(citation.source, 0) + 1
            
            # Count by categories
            for category in citation.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_citations': total_citations,
            'source_breakdown': source_counts,
            'top_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'average_relevance': sum(c.relevance_score for c in self.citations.values()) / total_citations if total_citations > 0 else 0
        }
    
    def export_bibliography(self, citation_ids: List[str], style: str = 'apa') -> str:
        """Export a bibliography for given citation IDs"""
        bibliography = []
        
        for citation_id in citation_ids:
            formatted_citation = self.format_citation(citation_id, style)
            bibliography.append(formatted_citation)
        
        return "\n\n".join(bibliography)
    
    def clear_citations(self):
        """Clear all citations"""
        self.citations.clear()
        self.save_citations()
        logger.info("Cleared all citations")