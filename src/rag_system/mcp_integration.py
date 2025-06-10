import asyncio
import json
import subprocess
from typing import Dict, List, Any, Optional
from .rag_engine import RAGEngine
from .conversation_memory import ConversationMemory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPIntegration:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.active_servers = {}
        self.conversation_memory = ConversationMemory()
    
    async def start_mcp_server(self, server_name: str, server_path: str) -> bool:
        """Start an MCP server"""
        try:
            if server_name in self.active_servers:
                logger.info(f"MCP server {server_name} already running")
                return True
            
            # Start the server process
            process = await asyncio.create_subprocess_exec(
                "python", server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.active_servers[server_name] = {
                'process': process,
                'path': server_path
            }
            
            logger.info(f"Started MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting MCP server {server_name}: {e}")
            return False
    
    async def stop_mcp_server(self, server_name: str) -> bool:
        """Stop an MCP server"""
        try:
            if server_name not in self.active_servers:
                logger.warning(f"MCP server {server_name} not running")
                return False
            
            process = self.active_servers[server_name]['process']
            process.terminate()
            await process.wait()
            
            del self.active_servers[server_name]
            logger.info(f"Stopped MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping MCP server {server_name}: {e}")
            return False
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on an MCP server"""
        try:
            if server_name not in self.active_servers:
                logger.error(f"MCP server {server_name} not running")
                return None
            
            process = self.active_servers[server_name]['process']
            
            # Prepare the tool call request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send request to server
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Read response
            response_line = await process.stdout.readline()
            response = json.loads(response_line.decode())
            
            if "error" in response:
                logger.error(f"MCP tool call error: {response['error']}")
                return None
            
            return response.get("result")
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name} on {server_name}: {e}")
            return None
    
    async def search_and_index_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv and add results to knowledge base"""
        try:
            # Call arXiv MCP server
            result = await self.call_mcp_tool(
                "arxiv-server",
                "search_arxiv",
                {
                    "query": query,
                    "max_results": max_results
                }
            )
            
            if not result:
                return []
            
            # Parse the result
            papers = json.loads(result[0]['text']) if result else []
            
            # Add to knowledge base
            if papers:
                await self.rag_engine.add_documents_from_source("arxiv", papers)
            
            logger.info(f"Indexed {len(papers)} arXiv papers for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching and indexing arXiv: {e}")
            return []
    
    async def search_and_index_github(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub and add results to knowledge base"""
        try:
            # Call GitHub MCP server
            result = await self.call_mcp_tool(
                "github-server",
                "search_github_repos",
                {
                    "query": query,
                    "per_page": max_results
                }
            )
            
            if not result:
                return []
            
            # Parse the result
            repos = json.loads(result[0]['text']) if result else []
            
            # Add to knowledge base
            if repos:
                await self.rag_engine.add_documents_from_source("github", repos)
            
            logger.info(f"Indexed {len(repos)} GitHub repositories for query: {query}")
            return repos
            
        except Exception as e:
            logger.error(f"Error searching and indexing GitHub: {e}")
            return []
    
    async def search_and_index_wikipedia(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia and add results to knowledge base"""
        try:
            # First search for articles
            search_result = await self.call_mcp_tool(
                "wikipedia-server",
                "search_wikipedia",
                {
                    "query": query,
                    "results": max_results
                }
            )
            
            if not search_result:
                return []
            
            # Parse search results
            article_titles = json.loads(search_result[0]['text']) if search_result else []
            
            # Get full articles
            articles = []
            for title in article_titles:
                article_result = await self.call_mcp_tool(
                    "wikipedia-server",
                    "get_wikipedia_summary",
                    {"title": title}
                )
                
                if article_result:
                    article_data = json.loads(article_result[0]['text'])
                    articles.append(article_data)
            
            # Add to knowledge base
            if articles:
                await self.rag_engine.add_documents_from_source("wikipedia", articles)
            
            logger.info(f"Indexed {len(articles)} Wikipedia articles for query: {query}")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching and indexing Wikipedia: {e}")
            return []
    
    async def multi_source_search(self, query: str, sources: List[str] = None, max_results_per_source: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Search multiple sources and index results"""
        if sources is None:
            sources = ["arxiv", "github", "wikipedia"]
        
        results = {}
        
        if "arxiv" in sources:
            results["arxiv"] = await self.search_and_index_arxiv(query, max_results_per_source)
        
        if "github" in sources:
            results["github"] = await self.search_and_index_github(query, max_results_per_source)
        
        if "wikipedia" in sources:
            results["wikipedia"] = await self.search_and_index_wikipedia(query, max_results_per_source)
        
        return results
    
    async def enhanced_query(self, user_query: str, auto_search: bool = True, 
                           search_sources: List[str] = None, session_id: str = None) -> Dict[str, Any]:
        """Enhanced query that automatically searches relevant sources with conversation memory"""
        try:
            # Get conversation context if session exists
            conversation_context = ""
            if session_id:
                conversation_context = self.conversation_memory.get_context_for_session(session_id)
            
            # If auto_search is enabled, search relevant sources first
            search_results = {}
            if auto_search:
                # Determine relevant sources based on query
                if search_sources is None:
                    search_sources = self._determine_relevant_sources(user_query)
                
                # Search and index new content
                search_results = await self.multi_source_search(
                    user_query, 
                    sources=search_sources,
                    max_results_per_source=3
                )
                
                # Log what was found
                total_results = sum(len(results) for results in search_results.values())
                if total_results > 0:
                    logger.info(f"Found and indexed {total_results} new documents from {list(search_results.keys())}")
            
            # Enhance query with conversation context
            enhanced_query = user_query
            if conversation_context:
                enhanced_query = f"Given this conversation context:\n{conversation_context}\n\nUser question: {user_query}"
            
            # Generate response using RAG
            response_data = await self.rag_engine.generate_response(enhanced_query, include_citations=True)
            
            # Add conversation turn to memory if session exists
            if session_id:
                self.conversation_memory.add_turn(
                    session_id=session_id,
                    user_message=user_query,
                    assistant_response=response_data['response'],
                    context_used=[],  # Could be populated with relevant docs
                    citations=response_data.get('citation_ids', []),
                    response_quality=response_data.get('search_quality', 0.0),
                    search_query=user_query,
                    search_results_count=sum(len(results) for results in search_results.values())
                )
            
            # Add search information to response
            response_data['search_results'] = search_results
            response_data['session_id'] = session_id
            response_data['conversation_context_used'] = bool(conversation_context)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in enhanced query: {e}")
            error_response = {
                'response': f"I encountered an error processing your query: {str(e)}",
                'citations': [],
                'citation_ids': [],
                'relevant_docs': 0,
                'search_quality': 0,
                'search_results': {},
                'session_id': session_id,
                'conversation_context_used': False
            }
            return error_response
    
    def _determine_relevant_sources(self, query: str) -> List[str]:
        """Determine which sources are most relevant for a query"""
        query_lower = query.lower()
        sources = []
        
        # Keywords that suggest academic papers
        academic_keywords = ["paper", "research", "study", "academic", "journal", "publication", "arxiv"]
        if any(keyword in query_lower for keyword in academic_keywords):
            sources.append("arxiv")
        
        # Keywords that suggest code/software
        code_keywords = ["code", "repository", "github", "implementation", "library", "framework", "algorithm"]
        if any(keyword in query_lower for keyword in code_keywords):
            sources.append("github")
        
        # Keywords that suggest general knowledge
        general_keywords = ["what is", "explain", "definition", "history", "overview"]
        if any(keyword in query_lower for keyword in general_keywords):
            sources.append("wikipedia")
        
        # Default to all sources if no specific indicators
        if not sources:
            sources = ["arxiv", "github", "wikipedia"]
        
        return sources
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers"""
        status = {}
        for server_name, server_info in self.active_servers.items():
            process = server_info['process']
            status[server_name] = {
                'running': process.returncode is None,
                'path': server_info['path'],
                'pid': process.pid if process.returncode is None else None
            }
        return status
    
    # Conversation memory methods
    
    def create_conversation_session(self, user_id: str = None, title: str = None) -> str:
        """Create a new conversation session"""
        return self.conversation_memory.create_session(user_id, title)
    
    def get_conversation_history(self, session_id: str, last_n_turns: int = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        from dataclasses import asdict
        turns = self.conversation_memory.get_session_history(session_id, last_n_turns)
        return [asdict(turn) for turn in turns]
    
    def search_conversation_history(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        from dataclasses import asdict
        turns = self.conversation_memory.search_conversations(query, user_id)
        return [asdict(turn) for turn in turns]
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        from dataclasses import asdict
        sessions = self.conversation_memory.get_user_sessions(user_id, active_only)
        return [asdict(session) for session in sessions]
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        return self.conversation_memory.update_session_title(session_id, title)
    
    def delete_conversation_session(self, session_id: str) -> bool:
        """Delete a conversation session"""
        return self.conversation_memory.delete_session(session_id)
    
    def get_conversation_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Get analytics about conversations"""
        return self.conversation_memory.get_conversation_analytics(user_id)
    
    def export_conversation(self, session_id: str, format: str = 'json') -> str:
        """Export a conversation session"""
        return self.conversation_memory.export_session(session_id, format)