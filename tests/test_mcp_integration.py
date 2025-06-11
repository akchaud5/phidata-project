"""Tests for MCPIntegration class"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from rag_system.mcp_integration import MCPIntegration


class TestMCPIntegration:
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_init(self, mock_memory):
        """Test MCPIntegration initialization"""
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        assert integration.rag_engine == mock_rag_engine
        assert integration.active_servers == {}
        mock_memory.assert_called_once()
    
    @patch('rag_system.mcp_integration.asyncio.create_subprocess_exec')
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_start_mcp_server(self, mock_memory, mock_subprocess):
        """Test starting MCP server"""
        mock_process = Mock()
        mock_subprocess.return_value = mock_process
        
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        result = await integration.start_mcp_server("test-server", "/path/to/server.py")
        
        assert result is True
        assert "test-server" in integration.active_servers
        assert integration.active_servers["test-server"]["process"] == mock_process
        mock_subprocess.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_stop_mcp_server(self, mock_memory):
        """Test stopping MCP server"""
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock()
        
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        integration.active_servers["test-server"] = {"process": mock_process}
        
        result = await integration.stop_mcp_server("test-server")
        
        assert result is True
        assert "test-server" not in integration.active_servers
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_call_mcp_tool(self, mock_memory):
        """Test calling MCP tool"""
        mock_stdin = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            return_value=b'{"jsonrpc":"2.0","id":1,"result":[{"text":"test result"}]}\n'
        )
        
        mock_process = Mock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        integration.active_servers["test-server"] = {"process": mock_process}
        
        result = await integration.call_mcp_tool(
            "test-server", 
            "test_tool", 
            {"arg1": "value1"}
        )
        
        assert result == [{"text": "test result"}]
        mock_stdin.write.assert_called()
        mock_stdin.drain.assert_called()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_search_and_index_arxiv(self, mock_memory):
        """Test searching and indexing arXiv papers"""
        mock_rag_engine = Mock()
        mock_rag_engine.add_documents_from_source = AsyncMock()
        
        integration = MCPIntegration(mock_rag_engine)
        integration.call_mcp_tool = AsyncMock(
            return_value=[{"text": '[{"title": "Test Paper", "summary": "Test summary"}]'}]
        )
        
        results = await integration.search_and_index_arxiv("machine learning", 5)
        
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"
        integration.call_mcp_tool.assert_called_once_with(
            "arxiv-server",
            "search_arxiv",
            {"query": "machine learning", "max_results": 5}
        )
        mock_rag_engine.add_documents_from_source.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_search_and_index_github(self, mock_memory):
        """Test searching and indexing GitHub repos"""
        mock_rag_engine = Mock()
        mock_rag_engine.add_documents_from_source = AsyncMock()
        
        integration = MCPIntegration(mock_rag_engine)
        integration.call_mcp_tool = AsyncMock(
            return_value=[{"text": '[{"name": "test-repo", "description": "Test repo"}]'}]
        )
        
        results = await integration.search_and_index_github("python library", 5)
        
        assert len(results) == 1
        assert results[0]["name"] == "test-repo"
        integration.call_mcp_tool.assert_called_once_with(
            "github-server",
            "search_github_repos",
            {"query": "python library", "per_page": 5}
        )
        mock_rag_engine.add_documents_from_source.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_multi_source_search(self, mock_memory):
        """Test multi-source search"""
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        # Mock individual search methods
        integration.search_and_index_arxiv = AsyncMock(return_value=[{"type": "arxiv"}])
        integration.search_and_index_github = AsyncMock(return_value=[{"type": "github"}])
        integration.search_and_index_wikipedia = AsyncMock(return_value=[{"type": "wikipedia"}])
        
        results = await integration.multi_source_search("test query", ["arxiv", "github"])
        
        assert "arxiv" in results
        assert "github" in results
        assert "wikipedia" not in results
        integration.search_and_index_arxiv.assert_called_once()
        integration.search_and_index_github.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    @pytest.mark.asyncio
    async def test_enhanced_query(self, mock_memory):
        """Test enhanced query with auto-search"""
        mock_rag_engine = Mock()
        mock_rag_engine.generate_response = AsyncMock(return_value={
            'response': 'Test response',
            'citations': ['Citation 1'],
            'citation_ids': ['cit1'],
            'search_quality': 0.8
        })
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_context_for_session.return_value = ""
        mock_memory_instance.add_turn = Mock()
        
        integration = MCPIntegration(mock_rag_engine)
        integration.multi_source_search = AsyncMock(return_value={
            "arxiv": [{"title": "Test Paper"}]
        })
        
        result = await integration.enhanced_query(
            "What is machine learning?",
            auto_search=True,
            session_id="test_session"
        )
        
        assert result['response'] == 'Test response'
        assert 'search_results' in result
        assert result['session_id'] == 'test_session'
        integration.multi_source_search.assert_called_once()
        mock_rag_engine.generate_response.assert_called_once()
        mock_memory_instance.add_turn.assert_called_once()
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_determine_relevant_sources_academic(self, mock_memory):
        """Test determining relevant sources for academic queries"""
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        sources = integration._determine_relevant_sources("research paper on machine learning")
        assert "arxiv" in sources
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_determine_relevant_sources_code(self, mock_memory):
        """Test determining relevant sources for code queries"""
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        sources = integration._determine_relevant_sources("github repository implementation")
        assert "github" in sources
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_determine_relevant_sources_general(self, mock_memory):
        """Test determining relevant sources for general queries"""
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        sources = integration._determine_relevant_sources("what is artificial intelligence")
        assert "wikipedia" in sources
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_get_server_status(self, mock_memory):
        """Test getting server status"""
        mock_process = Mock()
        mock_process.returncode = None
        mock_process.pid = 12345
        
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        integration.active_servers = {
            "test-server": {
                "process": mock_process,
                "path": "/path/to/server.py"
            }
        }
        
        status = integration.get_server_status()
        
        assert "test-server" in status
        assert status["test-server"]["running"] is True
        assert status["test-server"]["pid"] == 12345
        assert status["test-server"]["path"] == "/path/to/server.py"
    
    @patch('rag_system.mcp_integration.ConversationMemory')
    def test_conversation_memory_methods(self, mock_memory):
        """Test conversation memory wrapper methods"""
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.create_session.return_value = "session123"
        mock_memory_instance.get_session_history.return_value = []
        mock_memory_instance.search_conversations.return_value = []
        mock_memory_instance.get_user_sessions.return_value = []
        
        mock_rag_engine = Mock()
        integration = MCPIntegration(mock_rag_engine)
        
        # Test session creation
        session_id = integration.create_conversation_session("user1", "Test Session")
        assert session_id == "session123"
        
        # Test getting history
        history = integration.get_conversation_history("session123")
        assert history == []
        
        # Test search
        search_results = integration.search_conversation_history("test query")
        assert search_results == []
        
        # Test getting user sessions
        sessions = integration.get_user_sessions("user1")
        assert sessions == []