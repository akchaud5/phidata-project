#!/usr/bin/env python3
"""
Intelligent Research Assistant - Graduate-level RAG system with MCP servers

This is the main entry point for the application.
Run different interfaces using command-line arguments.
"""

import argparse
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_fastapi():
    """Run the FastAPI web interface"""
    from src.web_interface.main import app
    import uvicorn
    
    print("🚀 Starting FastAPI web interface...")
    print("📖 Access the application at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_streamlit():
    """Run the Streamlit interface"""
    import subprocess
    
    print("🚀 Starting Streamlit interface...")
    print("📖 Access the application at: http://localhost:8501")
    
    subprocess.run([
        "streamlit", "run", 
        "src/web_interface/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def run_cli():
    """Run the CLI interface"""
    from src.rag_system.rag_engine import RAGEngine
    from src.rag_system.mcp_integration import MCPIntegration
    
    async def cli_main():
        print("🧠 Initializing Intelligent Research Assistant CLI...")
        
        # Initialize system
        rag_engine = RAGEngine()
        mcp_integration = MCPIntegration(rag_engine)
        
        # Start MCP servers
        print("📡 Starting MCP servers...")
        servers = [
            ("arxiv-server", "src/mcp_servers/arxiv_server.py"),
            ("github-server", "src/mcp_servers/github_server.py"),
            ("wikipedia-server", "src/mcp_servers/wikipedia_server.py")
        ]
        
        for server_name, server_path in servers:
            try:
                await mcp_integration.start_mcp_server(server_name, server_path)
                print(f"✅ Started {server_name}")
            except Exception as e:
                print(f"❌ Failed to start {server_name}: {e}")
        
        # Create a session
        session_id = mcp_integration.create_conversation_session(title="CLI Session")
        print(f"📝 Created conversation session: {session_id}")
        
        print("\n🎯 Welcome to the Intelligent Research Assistant!")
        print("💡 Type 'help' for commands, 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("📚 Research Assistant> ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.lower().startswith('index '):
                    # Handle indexing commands
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 3:
                        source = parts[1]
                        query = parts[2]
                        print(f"🔍 Indexing {source} documents for '{query}'...")
                        
                        try:
                            if source == 'arxiv':
                                results = await mcp_integration.search_and_index_arxiv(query, 5)
                            elif source == 'github':
                                results = await mcp_integration.search_and_index_github(query, 5)
                            elif source == 'wikipedia':
                                results = await mcp_integration.search_and_index_wikipedia(query, 5)
                            else:
                                print(f"❌ Unknown source: {source}")
                                continue
                            
                            print(f"✅ Indexed {len(results)} documents from {source}")
                        except Exception as e:
                            print(f"❌ Error indexing: {e}")
                    else:
                        print("❌ Usage: index <source> <query>")
                    continue
                
                elif user_input.lower() == 'stats':
                    # Show statistics
                    stats = rag_engine.get_knowledge_base_stats()
                    print("\n📊 Knowledge Base Statistics:")
                    print(f"📄 Total Documents: {stats.get('total_documents', 0)}")
                    
                    source_breakdown = stats.get('source_breakdown', {})
                    for source, count in source_breakdown.items():
                        print(f"   - {source}: {count}")
                    
                    print(f"💾 Storage: {stats.get('persist_directory', 'N/A')}")
                    print()
                    continue
                
                elif user_input.lower() == 'sessions':
                    # Show conversation sessions
                    sessions = mcp_integration.get_user_sessions(None, active_only=True)
                    print(f"\n💬 Active Sessions: {len(sessions)}")
                    for session in sessions[:5]:  # Show first 5
                        print(f"   - {session['id'][:8]}: {session['title']} ({session['total_turns']} turns)")
                    print()
                    continue
                
                elif not user_input:
                    continue
                
                # Process query
                print("🤔 Thinking...")
                response_data = await mcp_integration.enhanced_query(
                    user_query=user_input,
                    auto_search=True,
                    session_id=session_id
                )
                
                print(f"\n💭 {response_data['response']}")
                
                # Show citations if available
                citations = response_data.get('citations', [])
                if citations:
                    print("\n📚 Citations:")
                    for citation in citations[:3]:  # Show first 3
                        print(f"   {citation}")
                
                # Show search info
                search_results = response_data.get('search_results', {})
                if search_results:
                    total_new = sum(len(results) for results in search_results.values())
                    if total_new > 0:
                        print(f"\n🔍 Found {total_new} new documents from {list(search_results.keys())}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def print_help():
        print("""
🆘 Available Commands:
   help                    - Show this help message
   index <source> <query>  - Index documents (sources: arxiv, github, wikipedia)
   stats                   - Show knowledge base statistics
   sessions               - Show conversation sessions
   exit/quit/q            - Exit the application
   
🔍 Examples:
   index arxiv "machine learning transformers"
   index github "python web framework"
   index wikipedia "artificial intelligence"
   
💡 Just type your research question to get started!
        """)
    
    asyncio.run(cli_main())

def test_system():
    """Test the system components"""
    print("🧪 Testing Intelligent Research Assistant components...")
    
    # Test imports
    try:
        from src.rag_system.rag_engine import RAGEngine
        from src.rag_system.mcp_integration import MCPIntegration
        from src.rag_system.vector_store import VectorStore
        from src.rag_system.citation_tracker import CitationTracker
        from src.rag_system.semantic_search import SemanticSearchEngine
        from src.rag_system.conversation_memory import ConversationMemory
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test basic functionality
    try:
        print("🔧 Testing vector store...")
        vector_store = VectorStore(persist_directory="./test_data")
        print("✅ Vector store initialized")
        
        print("🔧 Testing citation tracker...")
        citation_tracker = CitationTracker(storage_path="./test_citations.json")
        print("✅ Citation tracker initialized")
        
        print("🔧 Testing semantic search...")
        semantic_search = SemanticSearchEngine()
        print("✅ Semantic search initialized")
        
        print("🔧 Testing conversation memory...")
        conv_memory = ConversationMemory(storage_path="./test_conversations.json")
        print("✅ Conversation memory initialized")
        
        print("\n🎉 All components tested successfully!")
        print("🚀 System is ready to use!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Research Assistant - Graduate-level RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run FastAPI web interface (default)
  python main.py --interface web    # Run FastAPI web interface
  python main.py --interface streamlit  # Run Streamlit interface
  python main.py --interface cli    # Run CLI interface
  python main.py --test            # Test system components
        """
    )
    
    parser.add_argument(
        '--interface', '-i',
        choices=['web', 'streamlit', 'cli'],
        default='web',
        help='Interface to run (default: web)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test system components'
    )
    
    args = parser.parse_args()
    
    print("🧠 Intelligent Research Assistant")
    print("=" * 50)
    
    if args.test:
        test_system()
    elif args.interface == 'web':
        run_fastapi()
    elif args.interface == 'streamlit':
        run_streamlit()
    elif args.interface == 'cli':
        run_cli()

if __name__ == "__main__":
    main()