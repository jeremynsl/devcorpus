# DevCorpus

### Web Scraper with RAG Chat Interface

<img src="devcorpus-small.png" alt="drawing" width="200"/>

DevCorpus is a powerful web scraping application that allows you to scrape entire documentation websites and chat with the scraped content using advanced RAG (Retrieval-Augmented Generation) capabilities. Built with Python and featuring a modern Gradio UI.

## Features

-  **Smart Web Scraping**
  - Recursive website scraping for entire documentation sites (No API key required)
  - Pre-loaded with [~500 software documentation site URLs](docs.md) in the repository
  - Intelligent rate limiting and user agent management
  - Advanced text extraction with boilerplate removal
  - Support for both single pages and entire domains
  - Respects robots.txt and sitemap.xml

-  **Vector Database Storage**
  - ChromaDB integration for efficient vector storage
  - Switchable embedding models via SentenceTransformers
  - Smart text chunking with configurable settings
  - Automatic document deduplication
  - Support for multiple collections
  - Option to save scraped text to a local .txt file (use case: can paste directly into LLM context)

-  **Advanced Chat Interface**
  - RAG-powered conversations with scraped content
  - Can chat with multiple collections at once
  - Support for multiple LLM providers via LiteLLM (Gemini, GPT, Claude, OpenRouter etc)
  - Real-time document retrieval and with re-ranking for better results
  - Dynamic RAG prompts depending on relevance of retrieved documents
  - Citation of sources in responses

-  **Plan Mode**
  - Experimental Two-phase planning and execution for complex queries
  - Step-by-step solution generation
  - Context-aware document retrieval for each step

-  **Modern Gradio UI**
  - User-friendly interface for scraping and chat
  - Real-time progress tracking
  - Collection management
  - Model selection
  - Embedding reference citations
  - All settings configurable via the UI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/webscraper-rag-chat.git
   cd webscraper-rag-chat
   ```

2. Install dependencies either globally or in a python virtual environment (python -m venv venv)
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys to .env-example file and rename it to .env.

4. Run the application:
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

The Settings UI page allows you to configure various settings for the application. Alternatively, you can edit the `scraper_config.json` file directly. Here's a comprehensive guide to all available settings:

### Basic Settings
- **Proxies**: Comma-separated list of proxy IPs for web scraping.
- **Rate Limit**: Number of requests per second allowed when scraping (default: 3). Adjust based on website requirements.
- **User Agent**: Browser identifier string used for web requests. Default is Chrome on Windows.
- **PyTorch Device**: Device to use for ML models ('cuda' for GPU, 'cpu' for CPU). Default is 'cpu'. Requires app restart to take effect.

### Embeddings Settings
- **Available Embedding Models**: List of embedding models that can be used for document vectorization. Supports Hugging Face URLs.
- **Default Embedding Model**: The primary model used for creating document embeddings.
- **Reranker Model**: Model used to re-rank search results for better accuracy. Supports Hugging Face URLs
- **Initial Retrieval Percentage**: Percentage of documents from each collection to retrieve in the initial search phase (0.0-1.0).
- **Minimum Initial Results**: Minimum number of documents to retrieve, regardless of percentage.
- **Maximum Initial Results**: Maximum number of documents to retrieve, regardless of percentage.
- **Similarity Threshold**: Similarity threshold filters documents to retrieve based on relevance (0.0-1.0).

### Chunking Settings
- **Chunk Size**: Target size for text chunks when splitting documents.
- **Maximum Chunk Size**: Absolute maximum size for any single chunk.
- **Chunk Overlap**: Number of tokens to overlap between consecutive chunks for context preservation.

### Chat Settings
- **Message History Size**: Number of previous messages to maintain in chat context.
- **Maximum Retries**: Number of retry attempts for failed LLM API calls.
- **Retry Base Delay**: Initial delay (in seconds) between retry attempts.
- **Retry Maximum Delay**: Maximum delay (in seconds) between retry attempts.
- **System Prompt**: Base prompt that defines the AI chat assistant's behavior.
- **RAG Prompt**: Fallback template for how the AI should use retrieved documents to answer questions.
- **RAG Prompt High Quality**: Prompt that is used when high-quality context is available, prefers RAG over internal weights.
- **RAG Prompt Low Quality**: Prompt that is used when low-quality context is available, prefers internal weights over RAG.
- **Available Chat Models**: List of LLM models available for chat (supports various providers via LiteLLM).
- **Default Chat Model**: The default LLM model to use for chat interactions.

All settings can be modified through the Settings UI page, and changes take effect immediately unless noted otherwise. The configuration is stored in `scraper_config.json` in the project root directory.

### UI Settings
#### Some settings only available through the UI tabs

- **Save Text Files**: Toggle to save scraped text to a local .txt file in addition to vector storage.
- **RPM**: Requests per minute (RPM) for LLM rate limiting.
- **Cluster Chunking**: Toggle to use more advanced cluster chunking for text splitting.
- **Plan Mode**: Toggle to enable the experimental plan mode for more complex queries.
- **Add Summaries**: Trigger to add LLM summaries to embedded document chunks.  Be careful as this can trigger many API calls!


## FAQ

### General Questions

**Q: Do I need any API keys to use this?**  
A: No API key is required for the basic web scraping functionality. However, to use the chat interface, you'll need an API key for your chosen LLM (e.g., Gemini, OpenRouter etc).

**Q: What makes this different from other web scrapers?**  
A: DevCorpus combines web scraping with a RAG-powered chat interface, allowing you to not just scrape documentation but also interact with it naturally. It also features smart text extraction, automatic chunking, and multi-collection search capabilities.

**Q: Can I scrape any website?**  
A: While technically possible, DevCorpus is optimized for documentation websites. It includes features like boilerplate removal and intelligent chunking specifically designed for technical documentation.

### Technical Questions

**Q: How does the chunking work?**  
A: DevCorpus uses smart chunking that preserves semantic boundaries in documentation. Instead of splitting text arbitrarily, it respects natural breaks like headers and paragraphs.

**Q: What embedding models are supported?**  
A: DevCorpus supports multiple embedding models including Hugging Face's models.  Some non-standard Hugging Face models are not supported. You can configure your preferred model in the settings.  By default some performant and fast models are pre-configured

**Q: How many results are returned per RAG search?**  
A: By default, DevCorpus returns 5 results per collection searched. So if you search across 3 collections, you'll get up to 15 results, ranked by relevance.

**Q: Are JavaScript (ie dynamically generated) pages scrapable?**  
A: Currently, DevCorpus only scrapes static HTML pages.  This should accomodate most documentation websites.  In the future, it could be interesting to extend to scrape JavaScript-based websites as well.

### Troubleshooting

**Q: What if I get rate limited while scraping?**  
A: DevCorpus includes built-in rate limiting and retry mechanisms. You can adjust these settings in the configuration file.  You may also want to use a proxies in case your IP is blocked or rate limited.

**Q: Why isn't the chat model responding?**  
A: First ensure you've selected a documentation source in the dropdown. Also verify that your LLM API key is properly configured in the `.env` file.


## Requirements

- Python 3.11.1+
- ChromaDB
- Gradio
- aiohttp
- LiteLLM (for LLM integration)
- Trafilatura (for text extraction)
- Additional dependencies in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.  Also definitely add your documentation URLs to the `docs.md` file.

## License

Apache 2.0

## Acknowledgments

- Built with [Gradio](https://gradio.app/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Text extraction using [Trafilatura](https://trafilatura.readthedocs.io/)
- LLM integration using [LiteLLM](https://docs.litellm.ai/docs/)
- Web scraping using [aiohttp](https://docs.aiohttp.org/en/stable/)
- Chunking implementation from [Brandon Starxel's excellent code](https://github.com/brandonstarxel/chunking_evaluation)
