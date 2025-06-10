import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import arxiv
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Server("arxiv-server")

class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client()
    
    async def search_papers(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """Search arXiv papers by query"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=getattr(arxiv.SortCriterion, sort_by.title(), arxiv.SortCriterion.Relevance)
            )
            
            papers = []
            for result in self.client.results(search):
                paper_data = {
                    "id": result.entry_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat() if result.updated else None,
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "doi": result.doi,
                    "journal_ref": result.journal_ref,
                    "primary_category": result.primary_category
                }
                papers.append(paper_data)
            
            return papers
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get specific paper by arXiv ID"""
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(self.client.results(search), None)
            
            if not result:
                return None
            
            return {
                "id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat() if result.updated else None,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "primary_category": result.primary_category
            }
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

arxiv_client = ArxivClient()

@app.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available arXiv resources"""
    return [
        Resource(
            uri="arxiv://search",
            name="ArXiv Paper Search",
            description="Search for academic papers on arXiv",
            mimeType="application/json"
        ),
        Resource(
            uri="arxiv://categories",
            name="ArXiv Categories",
            description="List of arXiv subject categories",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read arXiv resources"""
    if uri == "arxiv://categories":
        categories = {
            "cs": "Computer Science",
            "math": "Mathematics", 
            "physics": "Physics",
            "q-bio": "Quantitative Biology",
            "q-fin": "Quantitative Finance",
            "stat": "Statistics",
            "eess": "Electrical Engineering and Systems Science",
            "econ": "Economics"
        }
        return json.dumps(categories, indent=2)
    
    return "Resource not found"

@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available arXiv tools"""
    return [
        Tool(
            name="search_arxiv",
            description="Search arXiv papers by query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for arXiv papers"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort criterion: relevance, lastUpdatedDate, submittedDate",
                        "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                        "default": "relevance"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_arxiv_paper",
            description="Get specific arXiv paper by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "ArXiv paper ID (e.g., '2301.00001')"
                    }
                },
                "required": ["paper_id"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    if name == "search_arxiv":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        sort_by = arguments.get("sort_by", "relevance")
        
        papers = await arxiv_client.search_papers(query, max_results, sort_by)
        
        return [TextContent(
            type="text",
            text=json.dumps(papers, indent=2)
        )]
    
    elif name == "get_arxiv_paper":
        paper_id = arguments.get("paper_id", "")
        paper = await arxiv_client.get_paper_by_id(paper_id)
        
        if paper:
            return [TextContent(
                type="text",
                text=json.dumps(paper, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Paper with ID {paper_id} not found"
            )]
    
    return [TextContent(type="text", text="Tool not found")]

async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="arxiv-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())