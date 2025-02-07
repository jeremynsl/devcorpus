import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.hub"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._config"
)

import gradio as gr
from scraper_chat.logger.logging_config import configure_logging, logger
from scraper_chat.database.chroma_handler import ChromaHandler
from scraper_chat.chat.chat_interface import ChatInterface
from scraper_chat.scraper.scraper import scrape_recursive
from scraper_chat.chunking.chunking import ChunkingManager
from scraper_chat.core.llm_config import LLMConfig
import json
from typing import List
import asyncio
import logging
from scraper_chat.ui.gradio_settings import create_settings_tab
from typing import Tuple, AsyncGenerator
import re
from scraper_chat.embeddings.embeddings import EmbeddingManager
from scraper_chat.config.config import load_config, save_config, CONFIG_FILE
from scraper_chat.ui.gradio_css import gradio_css

configure_logging()

logging.info("Launching DevCorpus")
# Load config
logging.info("Loading config")
config = load_config(CONFIG_FILE)


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
    def __init__(self) -> None:
        self.chat_interface = None
        self.history = []
        self.current_collections = []
        self.current_excerpts = []  # to store retrieved references

    def format_plan_mode_references(self) -> str:
        """Return a fixed note indicating that references are disabled in Plan Mode."""
        return "📝 **Note:** References are disabled in Plan Mode."

    def refresh_databases(self, current_selections: List[str]) -> None:
        """Refresh the list of available databases while maintaining current selections"""
        db = ChromaHandler()
        collections = db.get_available_collections()

        # Check which collections have summaries
        collection_choices = []
        for collection in collections:
            results = db.get_all_documents(collection)
            has_summary = (
                results
                and results["metadatas"]
                and any(metadata.get("summary") for metadata in results["metadatas"])
            )

            # Format display name with dots instead of underscores
            display_name = collection.replace("_", ".")
            if has_summary:
                collection_choices.append((f"📝 {display_name}", collection))
            else:
                collection_choices.append((display_name, collection))

        # Keep only current selections that still exist
        valid_selections = [s for s in current_selections if s in collections]

        return gr.Dropdown(
            choices=collection_choices, value=valid_selections, multiselect=True
        ), "Collections refreshed"

    def get_formatted_collections(self) -> List[Tuple[str, str]]:
        """Get collection list with summary indicators for initial dropdown"""
        db = ChromaHandler()
        collections = db.get_available_collections()
        collection_choices = []

        for collection in collections:
            results = db.get_all_documents(collection)
            has_summary = (
                results
                and results["metadatas"]
                and any(metadata.get("summary") for metadata in results["metadatas"])
            )

            # Format display name with dots instead of underscores
            display_name = collection.replace("_", ".")
            if has_summary:
                collection_choices.append((f"📝 {display_name}", collection))
            else:
                collection_choices.append((display_name, collection))

        return collection_choices

    def delete_collection(
        self, collections_to_delete: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Delete selected collections and refresh the list"""
        if not collections_to_delete:
            return "Please select collections to delete", gr.Dropdown()

        db = ChromaHandler()
        success = []
        failed = []
        for collection in collections_to_delete:
            if db.delete_collection(collection):
                success.append(collection)
                # Remove from current selections
                self.current_collections = [
                    c for c in self.current_collections if c != collection
                ]
            else:
                failed.append(collection)

        # Get updated collection list
        collections = db.get_available_collections()

        # Build status message
        status_msg = []
        if success:
            status_msg.append(f"Successfully deleted: {', '.join(success)}")
        if failed:
            status_msg.append(f"Failed to delete: {', '.join(failed)}")

        collection_choices = []
        for collection in collections:
            results = db.get_all_documents(collection)
            has_summary = (
                results
                and results["metadatas"]
                and any(metadata.get("summary") for metadata in results["metadatas"])
            )

            # Format display name with dots instead of underscores
            display_name = collection.replace("_", ".")
            if has_summary:
                collection_choices.append((f"📝 {display_name}", collection))
            else:
                collection_choices.append((display_name, collection))

        return (
            "\n".join(status_msg) or "No collections deleted",
            gr.Dropdown(
                choices=collection_choices,
                value=self.current_collections,
                multiselect=True,
            ),
        )

    def format_all_references(self, excerpts) -> str:
        """Format all references into a single string"""
        if not excerpts:
            return "No references available for this response."

        formatted_refs = []
        for i, excerpt in enumerate(excerpts, 1):
            # Build reference header
            ref_text = [f"**Reference [{i}]** from {excerpt['url']}"]

            # Add metadata
            ref_text.append(f"Relevance Score: {1 - excerpt['distance']:.2f}")
            if "metadata" in excerpt and excerpt["metadata"].get("summary"):
                ref_text.append(f"Summary: {excerpt['metadata']['summary']}")

            # Add separator and main text
            ref_text.extend(
                [
                    "",  # Empty line before content
                    excerpt["text"],
                    "-" * 80,  # Separator between references
                ]
            )

            formatted_refs.append("\n".join(ref_text))

        return "\n".join(formatted_refs)

    async def start_scraping(
        self,
        url: str,
        dump_text: bool,
        force_rescrape: bool,
        use_cluster: bool,
        progress: gr.HTML,
        tqdm_status: gr.Textbox,
    ) -> Tuple[str, str]:
        """Start the scraping process and update progress"""
        if not url.startswith(("http://", "https://")):
            yield "Error: Invalid URL. Must start with http:// or https://", ""
            return

        # Create handler for the Gradio progress box
        progress_handler = None
        monitor_task = None
        scrape_task = None

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
                        # Safely extract the tqdm line
                        parts = msg.split("scraper | ")
                        if len(parts) > 1:
                            self.tqdm_text = parts[1].strip()

                    # Add colored message to log
                    colored_msg = colorize_log(msg)
                    self.log_text += colored_msg

            # Set up handler
            progress_handler = ProgressHandler()
            progress_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

            # Add handler
            logging.getLogger().addHandler(progress_handler)

            # Create queue for progress updates
            queue = asyncio.Queue()

            # Create a task to monitor the log and update progress
            async def monitor_progress():
                last_length = 0
                last_tqdm = ""
                try:
                    while True:
                        current_text = progress_handler.log_text
                        current_tqdm = progress_handler.tqdm_text
                        if len(current_text) > last_length or current_tqdm != last_tqdm:
                            await queue.put((current_text[last_length:], current_tqdm))
                            last_length = len(current_text)
                            last_tqdm = current_tqdm
                        await asyncio.sleep(0.01)  # Check every 10ms
                except asyncio.CancelledError:
                    logger.debug("Monitor task cancelled")
                    raise

            # Start progress monitor
            monitor_task = asyncio.create_task(monitor_progress())

            # Load scraping config
            config = load_config(CONFIG_FILE)
            proxies = config.get("proxies", [])
            rate_limit = config.get("rate_limit", 5)
            user_agent = config.get("user_agent", "MyScraperBot/1.0")

            # Start scraper in background
            scrape_task = asyncio.create_task(
                scrape_recursive(url, user_agent, rate_limit, dump_text, force_rescrape)
            )

            # Stream progress updates while scraping runs
            try:
                while not scrape_task.done():
                    try:
                        new_text, tqdm_text = await asyncio.wait_for(
                            queue.get(), timeout=0.5
                        )
                        yield (
                            progress_handler.log_text,
                            f"Pages Scraped: {tqdm_text}"
                            if tqdm_text
                            else "Starting scrape...",
                        )
                    except asyncio.TimeoutError:
                        continue

                # Get final result and any remaining logs
                await scrape_task
                yield (
                    progress_handler.log_text
                    + '<span style="color: #00cc00">\nScraping completed successfully!</span>',
                    "Scraping completed",
                )

            except Exception as e:
                # Cancel tasks in case of error
                if not scrape_task.done():
                    scrape_task.cancel()
                    try:
                        await scrape_task
                    except asyncio.CancelledError:
                        pass
                yield (
                    f'<span style="color: #ff4444">Error during scraping: {str(e)}</span>',
                    "Error occurred",
                )

        except Exception as e:
            yield (
                f'<span style="color: #ff4444">Error: {str(e)}</span>',
                "Error occurred",
            )

        finally:
            # Clean up tasks and handlers
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            if scrape_task and not scrape_task.done():
                scrape_task.cancel()
                try:
                    await scrape_task
                except asyncio.CancelledError:
                    pass

            if progress_handler:
                root_logger = logging.getLogger()
                root_logger.removeHandler(progress_handler)

    async def chat(
        self, message: str, history: list, collections: list, model: str
    ) -> AsyncGenerator[Tuple[List[dict], str], None]:
        """Handle chat interaction with streaming"""
        if not self.chat_interface:
            history.append(
                {
                    "role": "assistant",
                    "content": "Please select a documentation source first.",
                }
            )
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
            async for chunk, excerpts in self.chat_interface.get_response(message):
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

    def initialize_chat(
        self, collections: list, model: str, rate_limit: int = 9
    ) -> Tuple[List[dict], str, str]:
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

    async def generate_summaries(
        self,
        collections: list,
        model: str,
        regenerate: bool = False,
        progress=gr.Progress(),
    ) -> str:
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
                if not docs or not docs["documents"]:
                    continue

                total_docs = len(docs["documents"])
                progress(0, desc=f"Processing {collection_name}")

                for i, (doc_id, text) in enumerate(zip(docs["ids"], docs["documents"])):
                    progress((i + 1) / total_docs)

                    # Skip if already has summary and not regenerating
                    if not regenerate and docs["metadatas"][i].get("summary"):
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
                            async for chunk, _ in self.chat_interface.get_response(
                                prompt
                            ):
                                chunk_text = (
                                    chunk["content"]
                                    if isinstance(chunk, dict)
                                    else chunk
                                )
                                summary_chunks.append(chunk_text)
                            summary = "".join(summary_chunks).strip()

                            # Verify we got a valid summary
                            if summary and not summary.startswith("Error"):
                                success = True
                                break

                        except Exception as e:
                            logger.error(
                                f"Error generating summary (attempt {retry + 1}) for {doc_id}: {str(e)}"
                            )
                            if retry < max_retries - 1:
                                await asyncio.sleep(
                                    retry_delay * (retry + 1)
                                )  # Exponential backoff
                            continue

                    if success:
                        # Update document metadata with summary
                        if self.chat_interface.db.update_document_metadata(
                            collection_name, doc_id, {"summary": summary}
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


async def process_chat(
    msg, hist, colls, mdl, pm, chat_app: GradioChat
) -> AsyncGenerator[Tuple[List[dict], str], None]:
    """
    Handle both Plan Mode and regular chat.

    Args:
        msg (str): User message.
        hist (list): Chat history.
        colls (list): Selected collections.
        mdl (str): Selected model.
        pm (bool): Plan Mode toggle.
        chat_app (GradioChat): Instance of GradioChat.

    Yields:
        tuple: (updated_history, references_text)
    """
    if not pm:
        # Regular Chat Mode
        if not chat_app.chat_interface:
            # Initialize chat interface if not already done
            chat_app.initialize_chat(colls, mdl)

        if not msg:
            hist.append({"role": "assistant", "content": "Please enter a message."})
            yield hist, ""
            return

        # Stream the response from regular chat
        async for updated_history, refs in chat_app.chat(msg, hist, colls, mdl):
            yield updated_history, refs
    else:
        # Plan Mode
        if not chat_app.chat_interface:
            # Initialize chat interface if not already done
            chat_app.initialize_chat(colls, mdl)

        if not msg:
            hist.append({"role": "assistant", "content": "Please enter a message."})
            yield hist, chat_app.format_plan_mode_references()
            return

        # Stream the response from Plan Mode via ChatInterface
        async for updated_history, refs in chat_app.chat_interface.plan_mode_chat(
            msg, hist, colls, mdl
        ):
            yield updated_history, refs


def load_doc_links():
    """Load documentation links from docs.md"""
    try:
        with open("docs.md", "r") as f:
            content = f.read()

        # Parse the markdown table using regex
        links = []
        current_category = None
        for line in content.split("\n"):
            if line.startswith("## "):
                current_category = line[3:].strip()
            elif (
                "|" in line and "[" in line and "]" in line and "http" in line
            ):  # Only match lines with URLs
                # Skip header and separator lines
                if "---" in line or "Project" in line:
                    continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:  # At least project and docs link
                    project = parts[0]
                    url_match = re.search(r"\((.*?)\)", parts[1])
                    if url_match:
                        url = url_match.group(1)
                        # Check if we have tags
                        tags = parts[2].split(",") if len(parts) > 2 else []
                        tags = [tag.strip() for tag in tags]
                        tag_text = f" [🏷️ {', '.join(tags)}]" if tags else ""
                        links.append(
                            f"{current_category} - {project}: {url} - {tag_text}"
                        )

        return [""] + sorted(links)  # Empty string for manual URL entry
    except Exception as e:
        logger.error(f"Error loading doc links: {e}")
        return [""]


def create_demo():
    """Create the Gradio demo Blocks layout."""
    # Load config for embedding models
    config = load_config(CONFIG_FILE)
    available_models = (
        config.get("embeddings", {}).get("models", {}).get("available", [])
    )
    default_model = config.get("embeddings", {}).get("models", {}).get("default")

    def update_embedding_model(model_name):
        try:
            # Update config
            config = load_config(CONFIG_FILE)
            config["embeddings"]["models"]["default"] = model_name
            save_config(config, CONFIG_FILE)

            # Reinitialize embedding manager
            EmbeddingManager._instance = None
            EmbeddingManager()

            return f"Successfully switched to embedding model: {model_name}"
        except Exception as e:
            return f"Error switching embedding model: {str(e)}"

    chat_app = GradioChat()

    with gr.Blocks(title="Documentation Chat & Scraper", css=gradio_css) as demo:
        # Configure queue with default settings
        demo.queue(default_concurrency_limit=1)

        with gr.Tabs():
            with gr.Tab("Scraper"):
                gr.Markdown(
                    """
                        ## Web Scraper
                        Enter a URL to scrape documentation from a website.
                        """
                )

                # Add dropdown for predefined documentation links
                doc_links = gr.Dropdown(
                    choices=load_doc_links(),
                    label="Select Documentation",
                    info="Choose from predefined documentation links or enter a custom URL below",
                )

                url = gr.Textbox(
                    label="URL", info="Enter the URL of the documentation to scrape"
                )

                # Add embedding model selector
                with gr.Row():
                    embedding_model = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Embedding Model",
                        info="Select the model to use for creating document embeddings",
                    )
                    embedding_status = gr.Textbox(
                        info="Status", label="Model Status", interactive=False
                    )

                # Connect embedding model change event
                embedding_model.change(
                    fn=update_embedding_model,
                    inputs=[embedding_model],
                    outputs=[embedding_status],
                )

                def update_url(selected):
                    if not selected:
                        return ""
                    # Get the part after the colon which contains the URL and the tags.
                    url_with_tags = selected.split(": ")[-1]
                    # Split again on ' - ' and take the first part (the URL).
                    url = url_with_tags.split(" - ")[0].strip()
                    return url

                # Update URL textbox when a documentation link is selected
                doc_links.change(fn=update_url, inputs=[doc_links], outputs=[url])

                with gr.Row():
                    dump_text = gr.Checkbox(
                        label="Save Text Files",
                        value=False,
                        info="Save the scraped content as raw text files",
                    )
                    force_rescrape = gr.Checkbox(
                        label="Force Rescrape",
                        value=False,
                        info="Rescrape pages even if they already exist in the database",
                    )
                    use_cluster = gr.Checkbox(
                        label="Use Cluster Chunking",
                        value=False,
                        info="More resource-intensive but better for retrieval",
                    )
                scrape_btn = gr.Button("Start Scraping", variant="primary", scale=1)

                tqdm_status = gr.Textbox(
                    label="Progress",
                    value="Ready to scrape...",
                    interactive=False,
                    show_label=False,
                    elem_classes="tqdm-status",
                )

                scrape_progress = gr.HTML(
                    # label="Scraping Progress",
                    value="Progress will appear here...",
                    show_label=False,
                    container=True,
                    elem_classes="scraping-progress",
                )

            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Row(elem_id="outer-row"):
                        # Left column: collections dropdown, then row of model settings
                        with gr.Column(elem_id="left-col"):
                            collections = gr.Dropdown(
                                choices=chat_app.get_formatted_collections(),
                                label="Select Documentation Sources",
                                multiselect=True,
                                value=[],
                                container=True,
                            )

                            with gr.Row():
                                model = gr.Dropdown(
                                    choices=config["chat"]["models"]["available"],
                                    value=config["chat"]["models"]["default"],
                                    label="Model",
                                    container=True,
                                )
                                rate_limit = gr.Number(
                                    value=9,
                                    label="API Rate Limit (RPM)",
                                    minimum=1,
                                    maximum=60,
                                    step=1,
                                )
                                plan_mode = gr.Checkbox(
                                    label="Plan Mode",
                                    value=False,
                                    info="Experimental Planning",
                                )

                        # Right column: status, then two rows of buttons/checkbox
                        with gr.Column(elem_id="right-col"):
                            status_text = gr.Textbox(
                                label="Status",
                                interactive=False,
                                container=True,
                                elem_id="status-text",
                            )

                            # First row of button + checkbox
                            with gr.Row(equal_height=True):
                                add_summaries_btn = gr.Button(
                                    "📝 Add Summaries",
                                    variant="secondary",
                                    min_width=300,
                                )
                                regenerate_summaries = gr.Checkbox(
                                    label="Regenerate Summaries",
                                    value=False,
                                    min_width=350,
                                )

                            # Second row of two buttons
                            with gr.Row(equal_height=True):
                                delete_btn = gr.Button(
                                    "🗑️ Delete Collection", variant="secondary"
                                )
                                refresh_btn = gr.Button(
                                    "🔄 Refresh Collections", variant="secondary"
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
                        sanitize_html=True,  # Safely render HTML/markdown
                    )

                with gr.Row():
                    message = gr.Textbox(
                        label="Your Question",
                        placeholder="Select documentation source(s) above, then ask a question...",
                        show_label=False,
                        container=False,
                        scale=8,
                    )
                    submit_btn = gr.Button(
                        "Send", variant="primary", scale=1, min_width=100
                    )

                # Create accordion but don't update its state
                accordion = gr.Accordion("References", open=False)
                with accordion:
                    references = gr.Markdown(
                        value="No references available yet.", show_label=False
                    )

                # Event handlers

                # Scraping events
                scrape_btn.click(
                    fn=chat_app.start_scraping,
                    inputs=[
                        url,
                        dump_text,
                        force_rescrape,
                        use_cluster,
                        scrape_progress,
                        tqdm_status,
                    ],
                    outputs=[scrape_progress, tqdm_status],
                    api_name="start_scraping",  # Add API name for better queue handling
                    show_progress=True,  # Show progress in the UI
                    concurrency_limit=1,  # Only allow one scraping job at a time
                )

                # Collection management events
                refresh_btn.click(
                    fn=chat_app.refresh_databases,
                    inputs=[collections],
                    outputs=[collections, status_text],
                    show_progress=False,
                )

                delete_btn.click(
                    fn=chat_app.delete_collection,
                    inputs=[collections],
                    outputs=[status_text, collections],
                    show_progress=True,
                ).then(
                    fn=lambda: None,  # Clear chat history after deletion
                    outputs=[chatbot],
                )

                # Database or LLM dropdown changes
                collections.change(
                    fn=chat_app.initialize_chat,
                    inputs=[collections, model, rate_limit],
                    outputs=[chatbot, status_text, references],
                )
                model.change(
                    fn=chat_app.initialize_chat,
                    inputs=[collections, model, rate_limit],
                    outputs=[chatbot, status_text, references],
                )
                rate_limit.change(
                    fn=chat_app.initialize_chat,
                    inputs=[collections, model, rate_limit],
                    outputs=[chatbot, status_text, references],
                )

                async def handle_submit(msg, hist, colls, mdl, pm):
                    """
                    Handle the submit action for both Plan Mode and regular chat.
                    """
                    # Process the chat and yield results
                    async for updated_history, refs in process_chat(
                        msg, hist, colls, mdl, pm, chat_app
                    ):
                        yield updated_history, refs

                submit_btn.click(
                    fn=handle_submit,
                    inputs=[message, chatbot, collections, model, plan_mode],
                    outputs=[chatbot, references],
                    queue=True,
                ).then(fn=lambda: "", outputs=message)

                # Send on ENTER
                message.submit(
                    fn=handle_submit,
                    inputs=[message, chatbot, collections, model, plan_mode],
                    outputs=[chatbot, references],
                    queue=True,
                ).then(fn=lambda: "", outputs=message)

                # Connect add summaries button
                add_summaries_btn.click(
                    fn=chat_app.generate_summaries,
                    inputs=[collections, model, regenerate_summaries],
                    outputs=[status_text],
                )

            with gr.Tab("Settings"):
                save_button, status_output = create_settings_tab()

                # When settings are saved, refresh the global config and UI components
                def refresh_app_config():
                    """Refresh the global config and update UI components"""
                    global config
                    with open("scraper_config.json", "r") as f:
                        config = json.load(f)

                    # Return updates for both chat and embedding model dropdowns
                    return [
                        gr.update(
                            choices=config["chat"]["models"]["available"],
                            value=config["chat"]["models"]["default"],
                        ),
                        gr.update(
                            choices=config["embeddings"]["models"]["available"],
                            value=config["embeddings"]["models"]["default"],
                        ),
                        "Settings updated - Models refreshed",
                    ]

                # Connect both save and refresh buttons to update components
                save_button.click(
                    fn=refresh_app_config,
                    inputs=[],
                    outputs=[model, embedding_model, status_output],
                ).then(  # Also reinitialize the chat interface
                    fn=chat_app.initialize_chat,
                    inputs=[collections, model, rate_limit],
                    outputs=[chatbot, status_text, references],
                )

        return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
