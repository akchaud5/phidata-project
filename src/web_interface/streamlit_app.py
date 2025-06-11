import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_system.rag_engine import RAGEngine
from rag_system.mcp_integration import MCPIntegration

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligent Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .source-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'mcp_integration' not in st.session_state:
    st.session_state.mcp_integration = None

@st.cache_resource
def initialize_system():
    """Initialize RAG engine and MCP integration with servers"""
    try:
        rag_engine = RAGEngine()
        mcp_integration = MCPIntegration(rag_engine)
        
        # Start MCP servers
        servers = [
            ("arxiv-server", "src/mcp_servers/arxiv_server.py"),
            ("github-server", "src/mcp_servers/github_server.py"),
            ("wikipedia-server", "src/mcp_servers/wikipedia_server.py")
        ]
        
        async def start_servers():
            for server_name, server_path in servers:
                try:
                    await mcp_integration.start_mcp_server(server_name, server_path)
                    print(f"✅ Started {server_name}")
                except Exception as e:
                    print(f"❌ Failed to start {server_name}: {e}")
        
        # Run server initialization
        asyncio.run(start_servers())
        
        return rag_engine, mcp_integration
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None

# Initialize system
if st.session_state.rag_engine is None:
    with st.spinner("Initializing Intelligent Research Assistant..."):
        st.session_state.rag_engine, st.session_state.mcp_integration = initialize_system()

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Intelligent Research Assistant</h1>
    <p>Graduate-level RAG system with MCP servers integration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📚 Document Sources")
    
    # Index new documents
    with st.expander("Index New Documents", expanded=True):
        source = st.selectbox(
            "Source",
            ["arxiv", "github", "wikipedia"],
            help="Choose the source to search and index"
        )
        
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., machine learning",
            help="Enter keywords to search for relevant documents"
        )
        
        max_results = st.slider(
            "Max Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of documents to index"
        )
        
        if st.button("🔍 Index Documents", type="primary"):
            if search_query and st.session_state.mcp_integration:
                with st.spinner(f"Searching {source} for '{search_query}'..."):
                    try:
                        if source == "arxiv":
                            results = asyncio.run(
                                st.session_state.mcp_integration.search_and_index_arxiv(
                                    search_query, max_results
                                )
                            )
                        elif source == "github":
                            results = asyncio.run(
                                st.session_state.mcp_integration.search_and_index_github(
                                    search_query, max_results
                                )
                            )
                        elif source == "wikipedia":
                            results = asyncio.run(
                                st.session_state.mcp_integration.search_and_index_wikipedia(
                                    search_query, max_results
                                )
                            )
                        
                        st.success(f"✅ Indexed {len(results)} documents from {source}")
                        
                        # Show preview of indexed documents
                        if results:
                            st.subheader("📄 Indexed Documents Preview")
                            for i, doc in enumerate(results[:3]):  # Show first 3
                                with st.expander(f"{i+1}. {doc.get('title', 'Untitled')[:50]}..."):
                                    if source == "arxiv":
                                        st.write(f"**Authors:** {', '.join(doc.get('authors', []))}")
                                        st.write(f"**Categories:** {', '.join(doc.get('categories', []))}")
                                        st.write(f"**Published:** {doc.get('published', 'N/A')}")
                                    elif source == "github":
                                        st.write(f"**Language:** {doc.get('language', 'N/A')}")
                                        st.write(f"**Stars:** {doc.get('stars', 0)}")
                                        st.write(f"**Topics:** {', '.join(doc.get('topics', []))}")
                                    elif source == "wikipedia":
                                        st.write(f"**URL:** {doc.get('url', 'N/A')}")
                                    
                                    summary = doc.get('summary', doc.get('description', ''))
                                    if summary:
                                        st.write(f"**Summary:** {summary[:200]}...")
                        
                    except Exception as e:
                        st.error(f"Error indexing documents: {e}")
            else:
                st.warning("Please enter a search query")
    
    # Upload custom document
    with st.expander("Upload Custom Document"):
        doc_title = st.text_input("Document Title")
        doc_content = st.text_area("Document Content", height=100)
        doc_source = st.text_input("Source/Category", value="upload")
        
        if st.button("📤 Upload Document"):
            if doc_title and doc_content and st.session_state.rag_engine:
                try:
                    documents = [{
                        "title": doc_title,
                        "content": doc_content,
                        "metadata": {"source": doc_source}
                    }]
                    
                    chunk_ids = asyncio.run(
                        st.session_state.rag_engine.add_documents_from_source(
                            doc_source, documents
                        )
                    )
                    
                    st.success(f"✅ Uploaded document with {len(chunk_ids)} chunks")
                except Exception as e:
                    st.error(f"Error uploading document: {e}")
            else:
                st.warning("Please fill in all fields")
    
    # Knowledge base statistics
    st.header("📊 Knowledge Base Stats")
    
    if st.session_state.rag_engine:
        try:
            stats = st.session_state.rag_engine.get_knowledge_base_stats()
            
            # Display stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_documents', 0)}</div>
                    <div class="stat-label">Total Documents</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                source_breakdown = stats.get('source_breakdown', {})
                arxiv_count = source_breakdown.get('arxiv', 0)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{arxiv_count}</div>
                    <div class="stat-label">arXiv Papers</div>
                </div>
                """, unsafe_allow_html=True)
            
            github_count = source_breakdown.get('github', 0)
            wikipedia_count = source_breakdown.get('wikipedia', 0)
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{github_count}</div>
                    <div class="stat-label">GitHub Repos</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{wikipedia_count}</div>
                    <div class="stat-label">Wikipedia</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all sources
            if source_breakdown:
                st.subheader("📂 Sources")
                for source, count in source_breakdown.items():
                    st.markdown(f"""
                    <span class="source-tag">{source}: {count}</span>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    
    # Clear knowledge base
    st.header("⚠️ Danger Zone")
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.rag_engine:
            try:
                success = asyncio.run(st.session_state.rag_engine.clear_knowledge_base())
                if success:
                    st.success("✅ Knowledge base cleared")
                    st.session_state.chat_history = []
                else:
                    st.error("Failed to clear knowledge base")
            except Exception as e:
                st.error(f"Error clearing knowledge base: {e}")

# Main chat interface
st.header("💬 Chat with Your Research Assistant")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)

# Chat input
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_query = st.text_input(
        "Ask me anything about research topics, papers, code...",
        key="chat_input",
        placeholder="e.g., What are the latest developments in transformer architectures?"
    )

with col2:
    auto_search = st.checkbox("Auto-search", value=True, help="Automatically search relevant sources")

with col3:
    if st.button("Send", type="primary"):
        if user_query and st.session_state.mcp_integration:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_query,
                'timestamp': datetime.now()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(
                        st.session_state.mcp_integration.enhanced_query(
                            user_query=user_query,
                            auto_search=auto_search
                        )
                    )
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now()
                    })
                    
                    # Rerun to display new messages
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# Advanced search options
with st.expander("🔧 Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Search Sources")
        search_arxiv = st.checkbox("arXiv Papers", value=True)
        search_github = st.checkbox("GitHub Repositories", value=True)
        search_wikipedia = st.checkbox("Wikipedia Articles", value=True)
    
    with col2:
        st.subheader("Response Settings")
        max_context = st.slider("Max Context Documents", 1, 10, 5)
        include_citations = st.checkbox("Include Citations", value=True)

# Quick actions
st.header("🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📝 Latest ML Papers"):
        quick_query = "What are the latest machine learning papers and their key contributions?"
        st.session_state.chat_history.append({'role': 'user', 'content': quick_query, 'timestamp': datetime.now()})
        st.rerun()

with col2:
    if st.button("💻 Popular ML Repos"):
        quick_query = "Show me popular machine learning repositories on GitHub with their descriptions"
        st.session_state.chat_history.append({'role': 'user', 'content': quick_query, 'timestamp': datetime.now()})
        st.rerun()

with col3:
    if st.button("🔬 Research Trends"):
        quick_query = "What are the current trends in AI and machine learning research?"
        st.session_state.chat_history.append({'role': 'user', 'content': quick_query, 'timestamp': datetime.now()})
        st.rerun()

with col4:
    if st.button("📚 Explain Concepts"):
        quick_query = "Explain transformer architecture and its applications in detail"
        st.session_state.chat_history.append({'role': 'user', 'content': quick_query, 'timestamp': datetime.now()})
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🧠 Intelligent Research Assistant - Powered by Phidata, MCP Servers, and RAG Technology</p>
    <p>Built for graduate-level research and analysis</p>
</div>
""", unsafe_allow_html=True)