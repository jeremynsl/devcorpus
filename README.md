# Web Scraper with RAG Chat Interface

A powerful web scraping application that allows you to scrape websites and chat with the scraped content using advanced RAG (Retrieval-Augmented Generation) capabilities. Built with Python and featuring a modern Gradio UI.

## Features

- 🌐 **Smart Web Scraping**
  - Recursive website scraping with configurable depth
  - Intelligent rate limiting and user agent management
  - Advanced text extraction with boilerplate removal
  - Support for both single pages and entire domains

- 💾 **Vector Database Storage**
  - ChromaDB integration for efficient vector storage
  - Smart text chunking with configurable settings
  - Automatic document deduplication
  - Support for multiple collections

- 🤖 **Advanced Chat Interface**
  - RAG-powered conversations with scraped content
  - Support for multiple LLM providers (Gemini, GPT, Claude)
  - Real-time document retrieval and reranking
  - Citation of sources in responses

- 🎯 **Plan Mode**
  - Two-phase planning and execution for complex queries
  - Step-by-step solution generation
  - Context-aware document retrieval for each step

- 🎨 **Modern Gradio UI**
  - User-friendly interface for scraping and chat
  - Real-time progress tracking
  - Collection management
  - Model selection
  - Reference viewing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/webscraper-rag-chat.git
   cd webscraper-rag-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a configuration file:
   - Copy `scraper_config.example.json` to `scraper_config.json`
   - Adjust settings as needed (rate limits, models, etc.)

## Usage

### Launch the Gradio Interface

```bash
python main.py
```

This will start the Gradio web interface where you can:
- Enter URLs to scrape
- Choose scraping and chunking settings
- Select collections to chat with
- Choose LLM models
- Toggle Plan Mode for complex queries


## Configuration

The `scraper_config.json` file allows you to configure:

- Scraping settings (rate limits, user agent)
- Chunking parameters
- Embedding models
- LLM models and settings
- RAG retrieval parameters
- Rate limiting
- Proxy settings
- Retry settings

Example configuration:
```json
{
    "proxies": [      
    ],
    "rate_limit": 3,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "embeddings": {
        "models": {
            "available": [
                "avsolatorio/GIST-Embedding-v0",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ],
            "default": "avsolatorio/GIST-Embedding-v0",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        "retrieval": {
            "initial_percent": 0.2,
            "min_initial": 50,
            "max_initial": 500
        }
    },
    "chunking": {
        "chunk_size": 200,
        "max_chunk_size": 200,
        "chunk_overlap": 0
    },
    "chat": {
        "message_history_size": 10,
        "max_retries": 3,
        "retry_base_delay": 1,
        "retry_max_delay": 30,
        "system_prompt": "You are a helpful AI assistant that helps users find and understand information from web pages.",
        "rag_prompt": "Use ONLY the following documentation excerpts to answer the question. If you cannot answer based on these excerpts, say so.\n\nDOCUMENTATION EXCERPTS:\n{context}\n\nUSER QUESTION: {query}\n\nPlease provide a clear and concise answer, citing specific sources with [number] format. If multiple sources support a point, cite all of them.\nIf you cannot answer the question based on the provided context, say so clearly.",
        "models": {
            "available": [
                "gemini/gemini-1.5-flash",              
                "gemini/gemini-exp-1206",
                "groq/llama-3.1-70b-versatile",                
                "claude-3"
            ],
            "default": "gemini/gemini-1.5-flash"
        }
    },
    "scraping": {        
        "chunk_size": 1000,
        "overlap": 200
    }
}
```

## Requirements

- Python 3.8+
- ChromaDB
- Gradio
- LiteLLM (for LLM integration)
- Trafilatura (for text extraction)
- Additional dependencies in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- Built with [Gradio](https://gradio.app/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Text extraction using [Trafilatura](https://trafilatura.readthedocs.io/)
