# gradio_app.py (or gradio.py)

import asyncio
import gradio as gr
from chat import ChatInterface
from logger import logger
from config import CONFIG_FILE, load_config
from scraper import scrape_recursive
import logging
from chroma import ChromaHandler
from typing import List
import json
import warnings
from chunking import ChunkingManager
from llm_config import LLMConfig

# Filter HF_HOME deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*HF_HOME.*')

# Load config
with open("scraper_config.json", "r") as f:
    config = json.load(f)

def colorize_log(record: str) -> str:
    """Add color to log messages based on level"""
    if "ERROR" in record or "Error" in record:
        return f'<span style="color: #ff4444">{record}</span>'
    elif "WARNING" in record:
        return f'<span style="color: #ffaa00">{record}</span>'
    elif "INFO" in record:
        return f'<span style="color: #00cc00">{record}</span>'
    elif "DEBUG" in record:
        return f'<span style="color: #888888">{record}</span>'
    return record

class GradioChat:
    def __init__(self):
        self.chat_interface = None
        self.history = []
        self.current_collections = []
        self.current_excerpts = []  # to store retrieved references
        
    def refresh_databases(self, current_selections: List[str]):
        """Refresh the list of available databases while maintaining current selections"""
        collections = ChromaHandler.get_available_collections()
        
        # Check which collections have summaries
        db = ChromaHandler()
        collection_choices = []
        for collection in collections:
            results = db.get_all_documents(collection)
            has_summary = results and results['metadatas'] and any(
                metadata.get('summary') for metadata in results['metadatas']
            )
            
            # Format display name with dots instead of underscores
            display_name = collection.replace("_", ".")
            if has_summary:
                collection_choices.append((f"📝 {display_name}", collection))
            else:
                collection_choices.append((display_name, collection))
        
        # Keep only current selections that still exist
        valid_selections = [s for s in current_selections if s in collections]
        
        return gr.Dropdown(choices=collection_choices, value=valid_selections, multiselect=True), "Collections refreshed"

    def get_formatted_collections(self):
        """Get collection list with summary indicators for initial dropdown"""
        collections = ChromaHandler.get_available_collections()
        db = ChromaHandler()
        collection_choices = []
        
        for collection in collections:
            results = db.get_all_documents(collection)
            has_summary = results and results['metadatas'] and any(
                metadata.get('summary') for metadata in results['metadatas']
            )
            
            # Format display name with dots instead of underscores
            display_name = collection.replace("_", ".")
            if has_summary:
                collection_choices.append((f"📝 {display_name}", collection))
            else:
                collection_choices.append((display_name, collection))
                
        return collection_choices

    def delete_collection(self, collections_to_delete: List[str]):
        """Delete selected collections and refresh the list"""
        if not collections_to_delete:
            return "Please select collections to delete", gr.Dropdown()
            
        success = []
        failed = []
        for collection in collections_to_delete:
            if ChromaHandler.delete_collection(collection):
                success.append(collection)
                # Remove from current selections
                self.current_collections = [c for c in self.current_collections if c != collection]
            else:
                failed.append(collection)
        
        # Get updated collection list
        available_collections = ChromaHandler.get_available_collections()
        
        # Build status message
        status_msg = []
        if success:
            status_msg.append(f"Successfully deleted: {', '.join(success)}")
        if failed:
            status_msg.append(f"Failed to delete: {', '.join(failed)}")
            
        return (
            "\n".join(status_msg) or "No collections deleted",
            gr.Dropdown(choices=available_collections, value=self.current_collections, multiselect=True)
        )

    def format_all_references(self, excerpts):
        """Format all references into a single string"""
        if not excerpts:
            return "No references available for this response."
            
        formatted_refs = []
        for i, excerpt in enumerate(excerpts, 1):
            # Build reference header
            ref_text = [f"**Reference [{i}]** from {excerpt['url']}"]
            
            # Add metadata
            ref_text.append(f"Relevance Score: {1 - excerpt['distance']:.2f}")
            if 'metadata' in excerpt and excerpt['metadata'].get('summary'):
                ref_text.append(f"Summary: {excerpt['metadata']['summary']}")
            
            # Add separator and main text
            ref_text.extend([
                "",  # Empty line before content
                excerpt['text'],
                "-" * 80  # Separator between references
            ])
            
            formatted_refs.append("\n".join(ref_text))
            
        return "\n".join(formatted_refs)

    async def start_scraping(self, url: str, store_db: bool, use_cluster: bool, progress: gr.HTML, tqdm_status: gr.Textbox):
        """Start the scraping process and update progress"""
        if not url.startswith(('http://', 'https://')):
            yield "Error: Invalid URL. Must start with http:// or https://", ""
            return
            
        try:
            # Set the chunking method based on checkbox - this affects ChromaHandler globally
            chunking_manager = ChunkingManager()
            if use_cluster:
                chunking_manager.use_cluster_chunker()
            else:
                chunking_manager.use_recursive_chunker()
                
            # Create handler for the Gradio progress box
            class ProgressHandler(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.log_text = ""
                    self.tqdm_text = ""
                    
                def emit(self, record):
                    msg = self.format(record) + "\n"
                    
                    # Extract tqdm output
                    if "Pages Scraped:" in msg and "pages/s" in msg:
                        # Get the full tqdm line without timestamp
                        self.tqdm_text = msg.split("RecursiveScraper | ")[1].strip()
                    
                    # Add colored message to log
                    colored_msg = colorize_log(msg)
                    self.log_text += colored_msg
                    
            # Set up handler
            progress_handler = ProgressHandler()
            progress_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            
            # Add handler
            logger.addHandler(progress_handler)
            
            # Create queue for progress updates
            queue = asyncio.Queue()
            
            # Create a task to monitor the log and update progress
            async def monitor_progress():
                last_length = 0
                last_tqdm = ""
                while True:
                    current_text = progress_handler.log_text
                    current_tqdm = progress_handler.tqdm_text
                    if len(current_text) > last_length or current_tqdm != last_tqdm:
                        await queue.put((current_text[last_length:], current_tqdm))
                        last_length = len(current_text)
                        last_tqdm = current_tqdm
                    await asyncio.sleep(0.1)  # Check every 100ms
            
            # Start progress monitor
            monitor_task = asyncio.create_task(monitor_progress())
            
            # Load scraping config
            proxies, rate_limit, user_agent = load_config(CONFIG_FILE)
            
            # Start scraper in background
            scrape_task = asyncio.create_task(
                scrape_recursive(url, user_agent, rate_limit, store_db)
            )
            
            # Stream progress updates while scraping runs
            try:
                while not scrape_task.done():
                    try:
                        new_text, tqdm_text = await asyncio.wait_for(queue.get(), timeout=0.5)
                        yield progress_handler.log_text, f"Pages Scraped: {tqdm_text}" if tqdm_text else "Starting scrape..."
                    except asyncio.TimeoutError:
                        continue
                        
                # Get final result and any remaining logs
                await scrape_task
                yield (
                    progress_handler.log_text + '<span style="color: #00cc00">\nScraping completed successfully!</span>',
                    "Scraping completed"
                )
                
            except Exception as e:
                # Cancel tasks in case of error
                if not scrape_task.done():
                    scrape_task.cancel()
                    try:
                        await scrape_task
                    except asyncio.CancelledError:
                        pass
                yield f'<span style="color: #ff4444">Error during scraping: {str(e)}</span>', "Error occurred"
                
            finally:
                # Clean up
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
                logger.removeHandler(progress_handler)
                # Clear the queue
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                
        except Exception as e:
            yield f'<span style="color: #ff4444">Error during scraping: {str(e)}</span>', "Error occurred"

    async def chat(self, message: str, history: list, collections: list, model: str):
        """Handle chat interaction with streaming"""
        if not self.chat_interface:
            history.append({"role": "assistant", "content": "Please select a documentation source first."})
            yield history, ""
            return
            
        if not message:
            history.append({"role": "assistant", "content": "Please enter a message."})
            yield history, ""
            return
            
        # Format user message
        history.append({"role": "user", "content": message})
        # Add empty assistant message
        history.append({"role": "assistant", "content": ""})
        
        # Initialize references display
        self.current_excerpts = []
        current_response = ""
        
        try:
            # Stream the response
            async for chunk, excerpts in self.chat_interface.get_response(message, return_excerpts=True):
                # Update references
                if excerpts and not self.current_excerpts:
                    self.current_excerpts = excerpts
                    references_text = self.format_all_references(excerpts)
                else:
                    references_text = self.format_all_references(self.current_excerpts)
                
                # Handle chunk whether it's a string or dict with content
                chunk_text = chunk["content"] if isinstance(chunk, dict) else chunk
                # Append to current response
                current_response += chunk_text
                # Update the last message (assistant's response)
                history[-1]["content"] = current_response
                
                yield history, references_text
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history[-1]["content"] = error_msg
            yield history, error_msg

    def initialize_chat(self, collections: list, model: str, rate_limit: int = 9):
        """Initialize (or switch) chat interface"""
        if not collections:
            return [], "Please select at least one documentation source.", ""
            
        # Configure rate limit
        LLMConfig.configure_rate_limit(rate_limit)
        
        # Create new chat interface
        self.chat_interface = ChatInterface(collections, model)
        self.current_collections = collections
        self.history = []
        self.current_excerpts = []
        
        return [], f"Chat initialized with: {', '.join(collections)}", ""

    async def generate_summaries(self, collections: list, model: str, regenerate: bool = False, progress=gr.Progress()) -> str:
        """Generate summaries for all documents in selected collections."""
        if not collections:
            return "Please select at least one collection."
            
        if not self.chat_interface:
            self.initialize_chat(collections, model)
            
        total_processed = 0
        total_updated = 0
        total_skipped = 0
        failed_docs = []
        max_retries = 3
        retry_delay = 5  # seconds
        
        try:
            for collection_name in collections:
                # Get all documents from collection
                docs = self.chat_interface.db.get_all_documents(collection_name)
                if not docs or not docs['documents']:
                    continue
                    
                total_docs = len(docs['documents'])
                progress(0, desc=f"Processing {collection_name}")
                
                for i, (doc_id, text) in enumerate(zip(docs['ids'], docs['documents'])):
                    progress((i + 1) / total_docs)
                    
                    # Skip if already has summary and not regenerating
                    if not regenerate and docs['metadatas'][i].get('summary'):
                        total_skipped += 1
                        total_processed += 1
                        continue
                    
                    # Generate summary using the chat model with retries
                    prompt = f"""Instruction:
                                Summarize the following text in one sentence, focusing on its key purpose, main idea, and unique value. Avoid unnecessary details and keep it concise. Do not include any preamble, such as "Here is a summarized answer."
                                
                                Example Input:
                                Svelte is a modern JavaScript framework that shifts the focus from runtime operations to compile-time transformations. By compiling components into highly efficient imperative code that updates the DOM directly, Svelte eliminates the need for a virtual DOM and reduces runtime overhead. It introduces unique features like reactive declarations, stores, and runes to handle state and reactivity declaratively. Compared to other frameworks, Svelte is lightweight, resulting in smaller bundle sizes and faster page loads. Additionally, its simple syntax and integration of CSS directly within components improve developer productivity and maintainability.

                                Example Output:
                                Svelte is a lightweight JavaScript framework that compiles components into efficient code, eliminating the virtual DOM for faster performance and offering features like reactive declarations and integrated CSS for simplicity.

                                Input:
                                {text}

                                Output:\n\n"""
                    summary = ""
                    success = False
                    
                    for retry in range(max_retries):
                        try:
                            summary_chunks = []
                            async for chunk, _ in self.chat_interface.get_response(prompt, return_excerpts=False):
                                chunk_text = chunk["content"] if isinstance(chunk, dict) else chunk
                                summary_chunks.append(chunk_text)
                            summary = "".join(summary_chunks).strip()
                            
                            # Verify we got a valid summary
                            if summary and not summary.startswith("Error"):
                                success = True
                                break
                            
                        except Exception as e:
                            logger.error(f"Error generating summary (attempt {retry + 1}) for {doc_id}: {str(e)}")
                            if retry < max_retries - 1:
                                await asyncio.sleep(retry_delay * (retry + 1))  # Exponential backoff
                            continue
                    
                    if success:
                        # Update document metadata with summary
                        if self.chat_interface.db.update_document_metadata(
                            collection_name, 
                            doc_id, 
                            {"summary": summary}
                        ):
                            total_updated += 1
                    else:
                        failed_docs.append(doc_id)
                    
                    total_processed += 1
            
            status = f"Processed {total_processed} documents. "
            if total_updated > 0:
                status += f"Added/updated {total_updated} summaries. "
            if total_skipped > 0:
                status += f"Skipped {total_skipped} existing summaries. "
            if failed_docs:
                status += f"\nFailed to generate summaries for {len(failed_docs)} documents after {max_retries} retries."
            return status
            
        except Exception as e:
            error_msg = f"Error generating summaries: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_demo():
    """Create the Gradio demo Blocks layout."""
    chat_app = GradioChat()

    with gr.Blocks(title="Documentation Chat & Scraper", css="""
        /* Supabase-inspired theme */
        :root {
            --background-color: #1c1c1c;
            --surface-color: #2a2a2a;
            --border-color: #404040;
            --text-color: #ffffff;
            --accent-color: #3ecf8e;  /* Supabase green */
            --error-color: #ff4444;
        }
        
        .gradio-container {
            background-color: var(--background-color) !important;
        }
        
        /* Make emojis more visible */
        .collection-emoji {
            font-size: 1.2em;
            margin-right: 0.5em;
            opacity: 1 !important;
        }
        
        /* Ensure dropdown text is visible */
        .gr-dropdown {
            color: var(--text-color) !important;
        }
        .gr-dropdown option {
            background-color: var(--surface-color);
            color: var(--text-color);
        }
        
        .tabs > .tab-nav {
            background-color: var(--background-color) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        
        .tabs > .tab-nav > button {
            color: var(--text-color) !important;
        }
        
        .tabs > .tab-nav > button.selected {
            color: var(--accent-color) !important;
            border-bottom-color: var(--accent-color) !important;
        }
        
        /* Input styling */
        input[type="text"], textarea {
            background-color: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Button styling */
        .primary {
            background-color: var(--accent-color) !important;
            color: var(--background-color) !important;
        }
        
        /* Progress display */
        .tqdm-status {
            background-color: transparent !important;
            border: none !important;
            color: var(--accent-color) !important;
            font-family: monospace;
            font-size: 0.9em;
            opacity: 0.8;            
            padding: 0 !important;
        }
        
        /* Log display */
        .scraping-progress {
            font-family: monospace;
            white-space: pre-wrap;
            
            height: 360px;
        }
        
        .scraping-progress::-webkit-scrollbar {
            width: 10px;
        }
        
        .scraping-progress::-webkit-scrollbar-track {
            background: var(--surface-color);
        }
        
        .scraping-progress::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 5px;
        }
        
        .scraping-progress::-webkit-scrollbar-thumb:hover {
            background: #666;
        }
        
        
        /* Reference link styling */
        .reference-link {
            color: var(--accent-color) !important;
            text-decoration: underline !important;
            font-weight: bold !important;
            padding: 0 2px !important;
            display: inline-block !important;
        }
        
        /* Markdown styling */
        .chat-window {
            font-family: system-ui, -apple-system, sans-serif;
        }
        
        .chat-window code {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 4px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        
        .chat-window pre {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            overflow-x: auto;
        }
        
        .chat-window pre code {
            background: none;
            border: none;
            padding: 0;
        }
        
        .chat-window blockquote {
            border-left: 3px solid var(--accent-color);
            margin: 8px 0;
            padding-left: 12px;
            color: #cccccc;
        }
        
        .chat-window table {
            border-collapse: collapse;
            margin: 8px 0;
            width: 100%;
        }
        
        .chat-window th,
        .chat-window td {
            border: 1px solid var(--border-color);
            padding: 6px 8px;
            text-align: left;
        }
        
        .chat-window th {
            background: var(--surface-color);
        }
        
        .chat-window {
            padding: 16px;
            background-color: var(--surface-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
    """) as demo:
        # Configure queue with default settings
        demo.queue(default_concurrency_limit=1)
        
        with gr.Tab("Scraper"):
            with gr.Row():
                url_input = gr.Textbox(
                    label="URL to Scrape",
                    placeholder="https://example.com/docs",
                    scale=4
                )
                store_db = gr.Checkbox(
                    label="Store in Database",
                    value=True,
                    info="Store in both text and ChromaDB (recommended)"
                )
                use_cluster = gr.Checkbox(
                    label="Use Cluster Chunking",
                    value=False,
                    info="Use semantic clustering for chunking (slower but more accurate)"
                )
                scrape_btn = gr.Button("Start Scraping", variant="primary", scale=1)
                
            tqdm_status = gr.Textbox(
                label="Progress",
                value="Ready to scrape...",
                interactive=False,
                show_label=False,
                elem_classes="tqdm-status"
            )
            
            scrape_progress = gr.HTML(
                #label="Scraping Progress",
                value="Progress will appear here...",
                show_label=False,
                container=True,
                elem_classes="scraping-progress",
            )
            
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    collections = gr.Dropdown(
                        choices=chat_app.get_formatted_collections(),  # Use formatted collections on initial load
                        label="Select Documentation Sources",
                        info="Choose one or more collections to search",
                        multiselect=True,
                        value=[],  # Start with no selection
                        container=True
                    )
                with gr.Column(scale=1):
                    model = gr.Dropdown(
                        choices=config["chat"]["models"]["available"],
                        value=config["chat"]["models"]["default"],
                        label="Model",
                        container=True,
                        scale=2
                    )
                    rate_limit = gr.Number(
                        value=9,
                        label="Rate Limit (RPM)",
                        info="API calls per minute",
                        minimum=1,
                        maximum=60,
                        step=1
                    )
                    with gr.Row():
                        add_summaries_btn = gr.Button("📝 Add Summaries", variant="secondary")
                        regenerate_summaries = gr.Checkbox(
                            label="Regenerate Existing",
                            value=False,
                            info="If checked, will regenerate existing summaries"
                        )
                with gr.Column(scale=1):
                    delete_btn = gr.Button("🗑️ Delete Collection", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Collections")
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        container=True,
                        #lines=10  # Make it bigger to show debug info
                    )
                    
                
               
            
            with gr.Row():
                chatbot = gr.Chatbot(
                    value=[],
                    type="messages",  # Use modern message format
                    label="Chat History",
                    height=400,
                    show_label=True,
                    container=True,
                    elem_classes="chat-window",
                    render_markdown=True,
                    layout="bubble",  # Better layout for markdown content
                    line_breaks=True,  # Preserve line breaks in messages
                    latex_delimiters=[  # Support LaTeX for math
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                    sanitize_html=True  # Safely render HTML/markdown
                )
            
            with gr.Row():
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Select documentation source(s) above, then ask a question...",
                    show_label=False,
                    container=False,
                    scale=8
                )
                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=100
                )
            
            # Create accordion but don't update its state
            accordion = gr.Accordion("References", open=False)
            with accordion:
                references = gr.Markdown(
                    value="No references available yet.",
                    show_label=False
                )
            
            # Event handlers
        
            # Scraping events
            scrape_btn.click(
                fn=chat_app.start_scraping,
                inputs=[
                    url_input,
                    store_db,
                    use_cluster,  # Add the new checkbox to inputs
                    scrape_progress,
                    tqdm_status
                ],
                outputs=[scrape_progress, tqdm_status],
                queue=True,  # Enable queue for streaming updates
                concurrency_limit=1  # Only allow one scraping job at a time
            )
        
            # Collection management events
            refresh_btn.click(
                fn=chat_app.refresh_databases,
                inputs=[collections],
                outputs=[collections, status_text],
                show_progress=False
            )
        
            delete_btn.click(
                fn=chat_app.delete_collection,
                inputs=[collections],
                outputs=[status_text, collections],
                show_progress=True
            ).then(
                fn=lambda: None,  # Clear chat history after deletion
                outputs=[chatbot]
            )

            # Database or LLM dropdown changes
            collections.change(
                fn=chat_app.initialize_chat,
                inputs=[collections, model, rate_limit],
                outputs=[chatbot, status_text, references]
            )
            model.change(
                fn=chat_app.initialize_chat,
                inputs=[collections, model, rate_limit],
                outputs=[chatbot, status_text, references]
            )
            rate_limit.change(
                fn=chat_app.initialize_chat,
                inputs=[collections, model, rate_limit],
                outputs=[chatbot, status_text, references]
            )

            # Send message (button)
            submit_btn.click(
                fn=chat_app.chat,
                inputs=[message, chatbot, collections, model],
                outputs=[chatbot, references],
                queue=True,
                concurrency_limit=None  # Allow unlimited chat concurrency since API handles rate limiting
            ).then(
                fn=lambda: "",
                outputs=message
            )

            # Send on ENTER
            message.submit(
                fn=chat_app.chat,
                inputs=[message, chatbot, collections, model],
                outputs=[chatbot, references],
                queue=True,
                concurrency_limit=None  # Allow unlimited chat concurrency since API handles rate limiting
            ).then(
                fn=lambda: "",
                outputs=message
            )

            # Connect add summaries button
            add_summaries_btn.click(
                fn=chat_app.generate_summaries,
                inputs=[collections, model, regenerate_summaries],
                outputs=[status_text],
            )
            
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
