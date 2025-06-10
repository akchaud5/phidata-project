# 🧠 Intelligent Research Assistant

A graduate-level research assistant system combining **MCP servers**, **Phidata**, and **RAG (Retrieval-Augmented Generation)** technology. This system automatically searches, indexes, and analyzes academic papers, code repositories, and other research materials to provide intelligent, context-aware responses with proper citations.

## ✨ Features

### 🔍 Multi-Source Research
- **arXiv Papers**: Automatic search and indexing of academic papers
- **GitHub Repositories**: Code analysis and repository insights
- **Wikipedia Articles**: General knowledge and background information
- **Custom Documents**: Upload your own research materials

### 🧠 Advanced AI Capabilities
- **Semantic Search**: Advanced embedding-based document retrieval
- **Hybrid Search**: Combines semantic and keyword search for optimal results
- **Citation Tracking**: Automatic citation generation in APA, MLA, and Chicago styles
- **Conversation Memory**: Maintains context across long research sessions

### 🖥️ Multiple Interfaces
- **Web Interface**: Modern, responsive web application with real-time chat
- **Streamlit App**: Interactive dashboard for research exploration
- **CLI Tool**: Command-line interface for power users

### 📊 Research Analytics
- Knowledge base statistics and insights
- Search quality metrics
- Conversation analytics
- Source breakdown and trending topics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akchaud5/phidata-project.git
   cd phidata-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   # Web interface (default)
   python main.py

   # Streamlit interface
   python main.py --interface streamlit

   # CLI interface
   python main.py --interface cli
   ```

### Environment Variables

Create a `.env` file with the following:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GITHUB_TOKEN=your_github_token_here

# Database Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
MAX_DOCUMENTS=10000
```

## 📖 Usage Guide

### Web Interface

1. **Start the server**: `python main.py`
2. **Open browser**: Navigate to `http://localhost:8000`
3. **Index sources**: Use the sidebar to search and index documents
4. **Ask questions**: Chat with the AI about your research topics

### Streamlit Interface

1. **Start Streamlit**: `python main.py --interface streamlit`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Interactive exploration**: Use the comprehensive dashboard

### CLI Interface

1. **Start CLI**: `python main.py --interface cli`
2. **Available commands**:
   - `help` - Show available commands
   - `index <source> <query>` - Index documents from a source
   - `stats` - Show knowledge base statistics
   - `sessions` - Show conversation sessions
   - Just type your question to get started!

### Example Workflows

#### Academic Research
```bash
# Index recent ML papers
index arxiv "transformer architecture 2024"

# Ask research questions
What are the latest developments in transformer architectures?
```

#### Code Research
```bash
# Index relevant repositories
index github "python machine learning library"

# Get implementation insights
How do popular ML libraries implement gradient descent?
```

#### Background Research
```bash
# Index background information
index wikipedia "artificial intelligence history"

# Build foundational knowledge
Explain the evolution of AI from symbolic systems to neural networks
```

## 🏗️ Architecture

```
├── src/
│   ├── mcp_servers/          # MCP server implementations
│   │   ├── arxiv_server.py   # arXiv paper retrieval
│   │   ├── github_server.py  # GitHub repository analysis
│   │   └── wikipedia_server.py # Wikipedia article fetching
│   ├── rag_system/           # Core RAG implementation
│   │   ├── rag_engine.py     # Main RAG orchestration
│   │   ├── vector_store.py   # Vector database management
│   │   ├── semantic_search.py # Advanced search capabilities
│   │   ├── citation_tracker.py # Citation management
│   │   ├── conversation_memory.py # Session management
│   │   ├── document_processor.py # Text processing
│   │   └── mcp_integration.py # MCP server integration
│   └── web_interface/        # User interfaces
│       ├── main.py          # FastAPI web app
│       └── streamlit_app.py # Streamlit dashboard
├── data/                    # Data storage
├── tests/                   # Test files
└── main.py                 # Application entry point
```

## 🔧 System Components

### MCP Servers
- **Modular Context Protocol**: Standardized way to connect to external data sources
- **Real-time Integration**: Dynamic content fetching and indexing
- **Extensible Design**: Easy to add new data sources

### RAG Engine
- **Phidata Integration**: Advanced RAG capabilities with conversation management
- **Vector Storage**: ChromaDB for efficient similarity search
- **Multiple LLMs**: Support for OpenAI GPT and Anthropic Claude

### Search & Retrieval
- **Semantic Search**: Sentence transformers for meaning-based retrieval
- **Keyword Search**: TF-IDF for exact term matching
- **Hybrid Approach**: Best of both semantic and keyword search

### Citation System
- **Automatic Tracking**: Every source automatically tracked
- **Multiple Formats**: APA, MLA, Chicago citation styles
- **Bibliography Export**: Generate properly formatted reference lists

## 🎯 Use Cases

### Graduate Students
- Literature reviews and paper analysis
- Research trend identification
- Citation management
- Cross-disciplinary research

### Researchers
- Rapid prototyping insights
- Code implementation examples
- Background research
- Collaborative knowledge building

### Educators
- Course material development
- Student research guidance
- Academic writing support
- Knowledge verification

## 🔄 Advanced Features

### Conversation Memory
- Maintains context across research sessions
- Session analytics and insights
- Export conversations in multiple formats
- Search through conversation history

### Semantic Analysis
- Document similarity detection
- Topic clustering and analysis
- Trend identification
- Related work discovery

### Quality Metrics
- Response quality scoring
- Source reliability assessment
- Search effectiveness tracking
- User satisfaction metrics

## 🧪 Testing

Run system tests:
```bash
python main.py --test
```

This validates:
- Component initialization
- API connectivity
- Database operations
- Search functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Phidata**: For the excellent RAG framework
- **MCP Protocol**: For standardized context integration
- **ChromaDB**: For vector storage capabilities
- **Sentence Transformers**: For semantic embeddings
- **FastAPI**: For the web framework
- **Streamlit**: For the interactive dashboard

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review example configurations

