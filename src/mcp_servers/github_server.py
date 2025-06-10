import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from github import Github
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Server("github-server")

class GitHubClient:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            logger.warning("GITHUB_TOKEN not set, some features may be limited")
        self.github = Github(token) if token else Github()
    
    async def search_repositories(self, query: str, sort: str = "stars", order: str = "desc", per_page: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        try:
            repos = self.github.search_repositories(
                query=query,
                sort=sort,
                order=order
            )
            
            results = []
            for i, repo in enumerate(repos):
                if i >= per_page:
                    break
                
                repo_data = {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "html_url": repo.html_url,
                    "clone_url": repo.clone_url,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "issues": repo.open_issues_count,
                    "created_at": repo.created_at.isoformat(),
                    "updated_at": repo.updated_at.isoformat(),
                    "topics": repo.get_topics(),
                    "license": repo.license.name if repo.license else None
                }
                results.append(repo_data)
            
            return results
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    async def get_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get detailed repository information"""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            
            return {
                "id": repository.id,
                "name": repository.name,
                "full_name": repository.full_name,
                "description": repository.description,
                "html_url": repository.html_url,
                "clone_url": repository.clone_url,
                "language": repository.language,
                "stars": repository.stargazers_count,
                "forks": repository.forks_count,
                "issues": repository.open_issues_count,
                "created_at": repository.created_at.isoformat(),
                "updated_at": repository.updated_at.isoformat(),
                "topics": repository.get_topics(),
                "license": repository.license.name if repository.license else None,
                "default_branch": repository.default_branch,
                "size": repository.size,
                "archived": repository.archived,
                "disabled": repository.disabled
            }
        except Exception as e:
            logger.error(f"Error fetching repository {owner}/{repo}: {e}")
            return None
    
    async def get_repository_contents(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """Get repository contents"""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            contents = repository.get_contents(path)
            
            if not isinstance(contents, list):
                contents = [contents]
            
            results = []
            for content in contents:
                content_data = {
                    "name": content.name,
                    "path": content.path,
                    "type": content.type,
                    "size": content.size,
                    "download_url": content.download_url,
                    "html_url": content.html_url
                }
                results.append(content_data)
            
            return results
        except Exception as e:
            logger.error(f"Error fetching contents for {owner}/{repo}/{path}: {e}")
            return []

github_client = GitHubClient()

@app.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available GitHub resources"""
    return [
        Resource(
            uri="github://search",
            name="GitHub Repository Search",
            description="Search for repositories on GitHub",
            mimeType="application/json"
        ),
        Resource(
            uri="github://trending",
            name="GitHub Trending",
            description="Trending repositories on GitHub",
            mimeType="application/json"
        )
    ]

@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available GitHub tools"""
    return [
        Tool(
            name="search_github_repos",
            description="Search GitHub repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for GitHub repositories"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort by: stars, forks, help-wanted-issues, updated",
                        "enum": ["stars", "forks", "help-wanted-issues", "updated"],
                        "default": "stars"
                    },
                    "order": {
                        "type": "string",
                        "description": "Order: asc, desc",
                        "enum": ["asc", "desc"],
                        "default": "desc"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page (max 100)",
                        "default": 10,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_github_repo",
            description="Get detailed information about a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner/organization"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    }
                },
                "required": ["owner", "repo"]
            }
        ),
        Tool(
            name="get_github_contents",
            description="Get contents of a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner/organization"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path within the repository (optional)",
                        "default": ""
                    }
                },
                "required": ["owner", "repo"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    if name == "search_github_repos":
        query = arguments.get("query", "")
        sort = arguments.get("sort", "stars")
        order = arguments.get("order", "desc")
        per_page = arguments.get("per_page", 10)
        
        repos = await github_client.search_repositories(query, sort, order, per_page)
        
        return [TextContent(
            type="text",
            text=json.dumps(repos, indent=2)
        )]
    
    elif name == "get_github_repo":
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        
        repo_info = await github_client.get_repository_info(owner, repo)
        
        if repo_info:
            return [TextContent(
                type="text",
                text=json.dumps(repo_info, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Repository {owner}/{repo} not found"
            )]
    
    elif name == "get_github_contents":
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        path = arguments.get("path", "")
        
        contents = await github_client.get_repository_contents(owner, repo, path)
        
        return [TextContent(
            type="text",
            text=json.dumps(contents, indent=2)
        )]
    
    return [TextContent(type="text", text="Tool not found")]

async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="github-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())