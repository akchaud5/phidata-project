import asyncio
import json
from typing import Dict, List, Optional, Any
import wikipedia
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Server("wikipedia-server")

class WikipediaClient:
    def __init__(self):
        wikipedia.set_lang("en")
    
    async def search_articles(self, query: str, results: int = 10) -> List[str]:
        """Search Wikipedia articles by query"""
        try:
            return wikipedia.search(query, results=results)
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    async def get_article_summary(self, title: str, sentences: int = 3) -> Optional[Dict[str, Any]]:
        """Get Wikipedia article summary"""
        try:
            summary = wikipedia.summary(title, sentences=sentences)
            page = wikipedia.page(title)
            
            return {
                "title": page.title,
                "summary": summary,
                "url": page.url,
                "categories": page.categories,
                "links": page.links[:20],  # Limit links to first 20
                "references": page.references[:10] if hasattr(page, 'references') else []
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Return the first option from disambiguation
            try:
                page = wikipedia.page(e.options[0])
                summary = wikipedia.summary(e.options[0], sentences=sentences)
                return {
                    "title": page.title,
                    "summary": summary,
                    "url": page.url,
                    "categories": page.categories,
                    "links": page.links[:20],
                    "references": page.references[:10] if hasattr(page, 'references') else [],
                    "disambiguation_options": e.options[:10]
                }
            except Exception as inner_e:
                logger.error(f"Error handling disambiguation for {title}: {inner_e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching Wikipedia article {title}: {e}")
            return None
    
    async def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full Wikipedia article content"""
        try:
            page = wikipedia.page(title)
            
            return {
                "title": page.title,
                "content": page.content,
                "summary": page.summary,
                "url": page.url,
                "categories": page.categories,
                "links": page.links,
                "references": page.references if hasattr(page, 'references') else [],
                "images": page.images[:5]  # Limit to first 5 images
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Return the first option from disambiguation
            try:
                page = wikipedia.page(e.options[0])
                return {
                    "title": page.title,
                    "content": page.content,
                    "summary": page.summary,
                    "url": page.url,
                    "categories": page.categories,
                    "links": page.links,
                    "references": page.references if hasattr(page, 'references') else [],
                    "images": page.images[:5],
                    "disambiguation_options": e.options[:10]
                }
            except Exception as inner_e:
                logger.error(f"Error handling disambiguation for {title}: {inner_e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching Wikipedia article content {title}: {e}")
            return None

wikipedia_client = WikipediaClient()

@app.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available Wikipedia resources"""
    return [
        Resource(
            uri="wikipedia://search",
            name="Wikipedia Article Search",
            description="Search for articles on Wikipedia",
            mimeType="application/json"
        ),
        Resource(
            uri="wikipedia://featured",
            name="Wikipedia Featured Articles",
            description="Featured articles on Wikipedia",
            mimeType="application/json"
        )
    ]

@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available Wikipedia tools"""
    return [
        Tool(
            name="search_wikipedia",
            description="Search Wikipedia articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Wikipedia articles"
                    },
                    "results": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 10)",
                        "default": 10,
                        "maximum": 50
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_wikipedia_summary",
            description="Get Wikipedia article summary",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Wikipedia article title"
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Number of sentences in summary (default: 3)",
                        "default": 3,
                        "maximum": 10
                    }
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="get_wikipedia_article",
            description="Get full Wikipedia article content",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Wikipedia article title"
                    }
                },
                "required": ["title"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    if name == "search_wikipedia":
        query = arguments.get("query", "")
        results = arguments.get("results", 10)
        
        articles = await wikipedia_client.search_articles(query, results)
        
        return [TextContent(
            type="text",
            text=json.dumps(articles, indent=2)
        )]
    
    elif name == "get_wikipedia_summary":
        title = arguments.get("title", "")
        sentences = arguments.get("sentences", 3)
        
        summary = await wikipedia_client.get_article_summary(title, sentences)
        
        if summary:
            return [TextContent(
                type="text",
                text=json.dumps(summary, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Wikipedia article '{title}' not found"
            )]
    
    elif name == "get_wikipedia_article":
        title = arguments.get("title", "")
        
        article = await wikipedia_client.get_article_content(title)
        
        if article:
            return [TextContent(
                type="text",
                text=json.dumps(article, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Wikipedia article '{title}' not found"
            )]
    
    return [TextContent(type="text", text="Tool not found")]

async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="wikipedia-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())