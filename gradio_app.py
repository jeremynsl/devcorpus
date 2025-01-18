# gradio_app.py (or gradio.py)

import asyncio
import gradio as gr
from chat import ChatInterface
from main import scrape_recursive, load_config, CONFIG_FILE, logger
import logging
from chroma import ChromaHandler
from typing import List
import json

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
        # Keep only current selections that still exist
        valid_selections = [s for s in current_selections if s in collections]
        return gr.Dropdown(choices=collections, value=valid_selections, multiselect=True), "Collections refreshed"

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
            formatted_refs.append(
                f"**Reference [{i}]** from {excerpt['url']}\n"
                f"Relevance Score: {1 - excerpt['distance']:.2f}\n\n"
                f"{excerpt['text']}\n"
                f"{'-' * 80}\n"  # Separator between references
            )
        return "\n".join(formatted_refs)

    async def start_scraping(self, url: str, store_db: bool, progress: gr.HTML, tqdm_status: gr.Textbox):
        """Start the scraping process and update progress"""
        if not url.startswith(('http://', 'https://')):
            yield "Error: Invalid URL. Must start with http:// or https://", ""
            return
            
        try:
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
                
            finally:
                # Clean up
                monitor_task.cancel()
                logger.removeHandler(progress_handler)
                
        except Exception as e:
            yield f'<span style="color: #ff4444">Error during scraping: {str(e)}</span>', "Error occurred"

    async def chat(self, message: str, history: list, collections: list, model: str):
        """Handle chat interaction with streaming"""
        try:
            # Initialize or update chat interface if collections changed
            if not self.chat_interface or set(collections) != set(self.current_collections):
                self.chat_interface = ChatInterface(collections, model)
                self.current_collections = collections

            # Add user message to history immediately
            history.append((message, ""))
            yield history, ""  # Show user message immediately
            
            # Start streaming response
            current_response = ""
            async for chunk, excerpts in self.chat_interface.get_response(message, return_excerpts=True):
                current_response += chunk
                # Update just the last response in history
                history[-1] = (message, current_response)
                yield history, self.format_all_references(excerpts)
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            history.append((message, f"Error: {str(e)}"))
            yield history, "Error occurred while processing your request."

    def initialize_chat(self, collections: list, model: str):
        """Initialize (or switch) chat interface"""
        if collections:
            self.chat_interface = ChatInterface(collections, model)
            self.current_collections = collections
        else:
            self.chat_interface = None
            self.current_collections = None
        self.history = []
        self.current_excerpts = []
        return [], f"Selected collections: {', '.join(collections) if collections else 'None'}", ""

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
            margin: -8px 0 8px 0;
            padding: 0 !important;
        }
        
        
        /* Log display */
        .scraping-progress {
            font-family: monospace;
            white-space: pre-wrap;
            padding: 16px;
            background-color: var(--surface-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            height: 400px;
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
        
        .chat-window {
            padding: 16px;
            background-color: var(--surface-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
    """) as demo:
        gr.Markdown("# Documentation Chat & Scraper")
        
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
                scrape_btn = gr.Button("Start Scraping", variant="primary", scale=1)
                
            tqdm_status = gr.Textbox(
                label="Progress",
                value="Ready to scrape...",
                interactive=False,
                show_label=False,
                elem_classes="tqdm-status"
            )
            
            scrape_progress = gr.HTML(
                label="Scraping Progress",
                value="Progress will appear here...",
                show_label=True,
                container=True,
                elem_classes="scraping-progress",
            )
            
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    collections = gr.Dropdown(
                        choices=ChromaHandler.get_available_collections(),
                        label="Select Documentation Sources",
                        info="Choose one or more collections to search",
                        multiselect=True,
                        value=[],  # Start with no selection
                        container=True
                    )
                model = gr.Dropdown(
                    choices=config["chat"]["models"]["available"],
                    value=config["chat"]["models"]["default"],
                    label="Model",
                    container=True,
                    scale=2
                     )
                with gr.Column(scale=1):
                    delete_btn = gr.Button("🗑️ Delete Collection", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Collections")
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        container=True
                    )
                    
                
               
            
            with gr.Row():
                chatbot = gr.Chatbot(
                    [],
                    label="Chat History",
                    height=400,
                    show_label=True,
                    container=True,
                    elem_classes="chat-window"
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
                inputs=[url_input, store_db, scrape_progress, tqdm_status],
                outputs=[scrape_progress, tqdm_status],
                queue=True  # Enable queuing for streaming
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
                inputs=[collections, model],
                outputs=[chatbot, gr.Markdown(), references],
            )
            model.change(
                fn=chat_app.initialize_chat,
                inputs=[collections, model],
                outputs=[chatbot, gr.Markdown(), references],
            )

            # Send message (button)
            submit_btn.click(
                fn=chat_app.chat,
                inputs=[message, chatbot, collections, model],
                outputs=[chatbot, references],  # Only update content, not accordion state
                queue=True
            ).then(
                fn=lambda: "",
                outputs=message
            )

            # Send on ENTER
            message.submit(
                fn=chat_app.chat,
                inputs=[message, chatbot, collections, model],
                outputs=[chatbot, references],  # Only update content, not accordion state
                queue=True
            ).then(
                fn=lambda: "",
                outputs=message
            )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
