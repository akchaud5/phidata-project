import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_system.rag_engine import RAGEngine
from rag_system.mcp_integration import MCPIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Research Assistant", description="Graduate-level RAG system with MCP servers")

@app.on_event("startup")
async def startup_event():
    """Initialize MCP servers on startup"""
    await start_mcp_servers()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine and MCP integration
rag_engine = RAGEngine()
mcp_integration = MCPIntegration(rag_engine)

# Global flag to track server initialization
mcp_servers_started = False

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now()
        }
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_id = self.connection_data.get(websocket, {}).get("client_id", "unknown")
            self.active_connections.remove(websocket)
            if websocket in self.connection_data:
                del self.connection_data[websocket]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove broken connections
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    auto_search: bool = True
    search_sources: Optional[List[str]] = None
    session_id: Optional[str] = None

class IndexRequest(BaseModel):
    source: str
    query: str
    max_results: int = 10

class DocumentUpload(BaseModel):
    title: str
    content: str
    source: str = "upload"
    metadata: Optional[Dict[str, Any]] = None

# API Routes
@app.get("/")
async def read_root():
    return HTMLResponse(content=get_html_content(), status_code=200)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with optional auto-search"""
    try:
        response = await mcp_integration.enhanced_query(
            user_query=request.query,
            auto_search=request.auto_search,
            search_sources=request.search_sources
        )
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def index_documents(request: IndexRequest):
    """Index documents from various sources"""
    try:
        if request.source == "arxiv":
            results = await mcp_integration.search_and_index_arxiv(
                request.query, request.max_results
            )
        elif request.source == "github":
            results = await mcp_integration.search_and_index_github(
                request.query, request.max_results
            )
        elif request.source == "wikipedia":
            results = await mcp_integration.search_and_index_wikipedia(
                request.query, request.max_results
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source: {request.source}")
        
        return {
            "indexed_count": len(results),
            "source": request.source,
            "query": request.query,
            "results": results[:5]  # Return first 5 for preview
        }
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(request: DocumentUpload):
    """Upload a custom document to the knowledge base"""
    try:
        documents = [{
            "title": request.title,
            "content": request.content,
            "metadata": request.metadata or {}
        }]
        
        chunk_ids = await rag_engine.add_documents_from_source(request.source, documents)
        
        return {
            "success": True,
            "chunk_ids": chunk_ids,
            "chunks_created": len(chunk_ids)
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = rag_engine.get_knowledge_base_stats()
        server_status = mcp_integration.get_server_status()
        
        return {
            "knowledge_base": stats,
            "mcp_servers": server_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear")
async def clear_knowledge_base():
    """Clear the knowledge base"""
    try:
        success = await rag_engine.clear_knowledge_base()
        return {"success": success}
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the query
            if message_data.get("type") == "query":
                query = message_data.get("message", "")
                auto_search = message_data.get("auto_search", True)
                search_sources = message_data.get("search_sources")
                
                # Send acknowledgment
                await manager.send_personal_message({
                    "type": "processing",
                    "message": "Processing your query...",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
                try:
                    # Generate response
                    response = await mcp_integration.enhanced_query(
                        user_query=query,
                        auto_search=auto_search,
                        search_sources=search_sources
                    )
                    
                    # Send response
                    await manager.send_personal_message({
                        "type": "response",
                        "message": response,
                        "original_query": query,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Error processing query: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message_data.get("type") == "index":
                source = message_data.get("source")
                query = message_data.get("query")
                max_results = message_data.get("max_results", 10)
                
                await manager.send_personal_message({
                    "type": "processing",
                    "message": f"Searching {source} for '{query}'...",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
                try:
                    # Index documents
                    if source == "arxiv":
                        results = await mcp_integration.search_and_index_arxiv(query, max_results)
                    elif source == "github":
                        results = await mcp_integration.search_and_index_github(query, max_results)
                    elif source == "wikipedia":
                        results = await mcp_integration.search_and_index_wikipedia(query, max_results)
                    else:
                        raise ValueError(f"Unsupported source: {source}")
                    
                    await manager.send_personal_message({
                        "type": "index_complete",
                        "message": f"Indexed {len(results)} documents from {source}",
                        "count": len(results),
                        "source": source,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Error indexing from {source}: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task to start MCP servers
async def start_mcp_servers():
    """Start all MCP servers"""
    global mcp_servers_started
    if mcp_servers_started:
        return
        
    servers = [
        ("arxiv-server", "src/mcp_servers/arxiv_server.py"),
        ("github-server", "src/mcp_servers/github_server.py"),
        ("wikipedia-server", "src/mcp_servers/wikipedia_server.py")
    ]
    
    logger.info("Starting MCP servers...")
    for server_name, server_path in servers:
        try:
            await mcp_integration.start_mcp_server(server_name, server_path)
            logger.info(f"✅ Started {server_name}")
        except Exception as e:
            logger.error(f"❌ Failed to start {server_name}: {e}")
    
    mcp_servers_started = True
    logger.info("MCP server initialization complete")

def get_html_content():
    """Return the HTML content for the web interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Research Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            color: white;
            font-size: 2rem;
            font-weight: 600;
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.8);
            margin-top: 0.5rem;
        }
        
        .main-container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            width: 100%;
            margin: 2rem auto;
            gap: 2rem;
            padding: 0 2rem;
        }
        
        .chat-container {
            flex: 2;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        
        .sidebar {
            flex: 1;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            height: fit-content;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 1rem;
            border-radius: 12px;
            line-height: 1.5;
        }
        
        .message.user {
            background: #667eea;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        
        .message.assistant {
            background: #f8f9fa;
            color: #333;
            border: 1px solid #e9ecef;
        }
        
        .message.processing {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            font-style: italic;
        }
        
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .chat-input {
            border-top: 1px solid #e9ecef;
            padding: 1.5rem;
            display: flex;
            gap: 1rem;
        }
        
        .chat-input input {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            font-size: 1rem;
        }
        
        .chat-input button {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
        }
        
        .chat-input button:hover {
            background: #5a6fd8;
        }
        
        .chat-input button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .sidebar h3 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        
        .index-form {
            margin-bottom: 2rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #555;
            font-weight: 500;
        }
        
        .form-group select,
        .form-group input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            width: 100%;
        }
        
        .btn:hover {
            background: #218838;
        }
        
        .btn.btn-danger {
            background: #dc3545;
        }
        
        .btn.btn-danger:hover {
            background: #c82333;
        }
        
        .stats {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .stats-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .disconnected {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="header">
        <h1>🧠 Intelligent Research Assistant</h1>
        <p>Graduate-level RAG system with MCP servers integration</p>
    </div>
    
    <div class="main-container">
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    Welcome! I'm your intelligent research assistant. I can help you search and analyze academic papers, code repositories, and other research materials. Ask me anything!
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ask me about research topics, papers, code..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>📚 Index Sources</h3>
            <div class="index-form">
                <div class="form-group">
                    <label>Source:</label>
                    <select id="indexSource">
                        <option value="arxiv">arXiv Papers</option>
                        <option value="github">GitHub Repositories</option>
                        <option value="wikipedia">Wikipedia Articles</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Search Query:</label>
                    <input type="text" id="indexQuery" placeholder="e.g., machine learning" />
                </div>
                <div class="form-group">
                    <label>Max Results:</label>
                    <input type="number" id="maxResults" value="5" min="1" max="20" />
                </div>
                <button class="btn" onclick="indexDocuments()">Index Documents</button>
            </div>
            
            <h3>📊 Statistics</h3>
            <div class="stats" id="stats">
                <div class="stats-item">
                    <span>Total Documents:</span>
                    <span id="totalDocs">-</span>
                </div>
                <div class="stats-item">
                    <span>arXiv Papers:</span>
                    <span id="arxivCount">-</span>
                </div>
                <div class="stats-item">
                    <span>GitHub Repos:</span>
                    <span id="githubCount">-</span>
                </div>
                <div class="stats-item">
                    <span>Wikipedia:</span>
                    <span id="wikipediaCount">-</span>
                </div>
            </div>
            
            <button class="btn btn-danger" onclick="clearKnowledgeBase()">Clear All Data</button>
        </div>
    </div>
    
    <script>
        let socket;
        let isConnected = false;
        const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function(event) {
                isConnected = true;
                updateConnectionStatus();
                console.log('WebSocket connected');
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            socket.onclose = function(event) {
                isConnected = false;
                updateConnectionStatus();
                console.log('WebSocket closed');
                // Attempt to reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateConnectionStatus() {
            const statusEl = document.getElementById('connectionStatus');
            if (isConnected) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status connected';
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status disconnected';
            }
        }
        
        function handleMessage(data) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageEl = document.createElement('div');
            
            switch(data.type) {
                case 'processing':
                    messageEl.className = 'message processing';
                    messageEl.textContent = data.message;
                    break;
                case 'response':
                    messageEl.className = 'message assistant';
                    messageEl.innerHTML = formatMessage(data.message);
                    break;
                case 'error':
                    messageEl.className = 'message error';
                    messageEl.textContent = data.message;
                    break;
                case 'index_complete':
                    messageEl.className = 'message assistant';
                    messageEl.textContent = `✅ ${data.message}`;
                    updateStats();
                    break;
                default:
                    messageEl.className = 'message assistant';
                    messageEl.textContent = data.message;
            }
            
            messagesContainer.appendChild(messageEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function formatMessage(message) {
            // Simple markdown-like formatting
            return message
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/\\n/g, '<br>');
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || !isConnected) return;
            
            // Add user message to chat
            const messagesContainer = document.getElementById('chatMessages');
            const userMessageEl = document.createElement('div');
            userMessageEl.className = 'message user';
            userMessageEl.textContent = message;
            messagesContainer.appendChild(userMessageEl);
            
            // Send to server
            socket.send(JSON.stringify({
                type: 'query',
                message: message,
                auto_search: true
            }));
            
            input.value = '';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function indexDocuments() {
            const source = document.getElementById('indexSource').value;
            const query = document.getElementById('indexQuery').value.trim();
            const maxResults = parseInt(document.getElementById('maxResults').value);
            
            if (!query || !isConnected) return;
            
            socket.send(JSON.stringify({
                type: 'index',
                source: source,
                query: query,
                max_results: maxResults
            }));
            
            // Add notification to chat
            const messagesContainer = document.getElementById('chatMessages');
            const notificationEl = document.createElement('div');
            notificationEl.className = 'message processing';
            notificationEl.textContent = `Indexing ${source} documents for "${query}"...`;
            messagesContainer.appendChild(notificationEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                const kb = data.knowledge_base;
                document.getElementById('totalDocs').textContent = kb.total_documents || 0;
                document.getElementById('arxivCount').textContent = kb.source_breakdown?.arxiv || 0;
                document.getElementById('githubCount').textContent = kb.source_breakdown?.github || 0;
                document.getElementById('wikipediaCount').textContent = kb.source_breakdown?.wikipedia || 0;
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        async function clearKnowledgeBase() {
            if (!confirm('Are you sure you want to clear all data? This cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch('/api/clear', { method: 'DELETE' });
                const data = await response.json();
                
                if (data.success) {
                    alert('Knowledge base cleared successfully!');
                    updateStats();
                } else {
                    alert('Failed to clear knowledge base.');
                }
            } catch (error) {
                alert('Error clearing knowledge base: ' + error.message);
            }
        }
        
        // Event listeners
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        document.getElementById('sendButton').addEventListener('click', sendMessage);
        
        // Initialize
        connectWebSocket();
        updateStats();
        
        // Update stats every 30 seconds
        setInterval(updateStats, 30000);
    </script>
</body>
</html>
    '''

if __name__ == "__main__":
    # Run the FastAPI app (MCP servers will start via startup event)
    uvicorn.run(app, host="0.0.0.0", port=8000)