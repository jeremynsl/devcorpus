import gradio as gr
import json
import os
import logging

logger = logging.getLogger(__name__)

def save_individual_settings(proxies_text, rate_limit_value):
    try:
        # Convert comma-separated string to list of IPs
        proxies = [p.strip() for p in proxies_text.split(',') if p.strip()]
        # Load current config and update with new settings
        config = load_config()
        config['proxies'] = proxies
        config['rate_limit'] = int(rate_limit_value)
        save_config(config)
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {str(e)}"

def load_config():
    # Compute the path to the scraper_config.json file relative to this file
    config_path = os.path.join(os.path.dirname(__file__), '../../scraper_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    else:
        return {}

def save_config(new_config):
    config_path = os.path.join(os.path.dirname(__file__), '../../scraper_config.json')
    with open(config_path, 'w') as f:
        json.dump(new_config, f, indent=4)


def save_settings(settings_text):
    try:
        settings = json.loads(settings_text)
        save_config(settings)
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {str(e)}"

def save_all_settings(proxies, rate_limit, user_agent,
                     embedding_models, default_embedding, reranker_model,
                     initial_percent, min_initial, max_initial,
                     chunk_size, max_chunk_size, chunk_overlap,
                     msg_history_size, max_retries, retry_base_delay, retry_max_delay,
                     system_prompt, rag_prompt,
                     chat_models, default_chat_model,
                     scraping_chunk_size, scraping_overlap,
                     pytorch_device):
    try:
        config = load_config()
        
        # Basic settings
        config['proxies'] = [p.strip() for p in proxies.split(',') if p.strip()]
        config['rate_limit'] = int(rate_limit)
        config['user_agent'] = user_agent
        config['pytorch_device'] = pytorch_device
        
        # Embeddings settings
        config['embeddings']['models']['available'] = [m.strip() for m in embedding_models.split(',')]
        config['embeddings']['models']['default'] = default_embedding
        config['embeddings']['models']['reranker'] = reranker_model
        config['embeddings']['retrieval']['initial_percent'] = float(initial_percent)
        config['embeddings']['retrieval']['min_initial'] = int(min_initial)
        config['embeddings']['retrieval']['max_initial'] = int(max_initial)
        
        # Chunking settings
        config['chunking']['chunk_size'] = int(chunk_size)
        config['chunking']['max_chunk_size'] = int(max_chunk_size)
        config['chunking']['chunk_overlap'] = int(chunk_overlap)
        
        # Chat settings 
        config['chat']['message_history_size'] = int(msg_history_size)
        config['chat']['max_retries'] = int(max_retries)
        config['chat']['retry_base_delay'] = float(retry_base_delay)
        config['chat']['retry_max_delay'] = float(retry_max_delay)
        config['chat']['system_prompt'] = system_prompt
        config['chat']['rag_prompt'] = rag_prompt
        config['chat']['models']['available'] = [m.strip() for m in chat_models.split(',')]
        config['chat']['models']['default'] = default_chat_model
        
        # Scraping settings
        config['scraping']['chunk_size'] = int(scraping_chunk_size)
        config['scraping']['overlap'] = int(scraping_overlap)
        
        save_config(config)
        return "Settings saved successfully. Please wait for UI refresh..."
    except Exception as e:
        return f"Error saving settings: {str(e)}"

def create_settings_tab():   
    current_config = load_config()
    
    def refresh_settings():
        """Refresh all settings from config file"""
        config = load_config()
        try:
            return [
                ', '.join(config.get('proxies', [])),
                int(config.get('rate_limit', 3)),
                config.get('user_agent', ''),
                ', '.join(config.get('embeddings', {}).get('models', {}).get('available', [])),
                config.get('embeddings', {}).get('models', {}).get('default', ''),
                config.get('embeddings', {}).get('models', {}).get('reranker', ''),
                float(config.get('embeddings', {}).get('retrieval', {}).get('initial_percent', 0.2)),
                int(config.get('embeddings', {}).get('retrieval', {}).get('min_initial', 50)),
                int(config.get('embeddings', {}).get('retrieval', {}).get('max_initial', 500)),
                int(config.get('chunking', {}).get('chunk_size', 200)),
                int(config.get('chunking', {}).get('max_chunk_size', 200)),
                int(config.get('chunking', {}).get('chunk_overlap', 0)),
                int(config.get('chat', {}).get('message_history_size', 10)),
                int(config.get('chat', {}).get('max_retries', 3)),
                float(config.get('chat', {}).get('retry_base_delay', 1)),
                float(config.get('chat', {}).get('retry_max_delay', 30)),
                config.get('chat', {}).get('system_prompt', ''),
                config.get('chat', {}).get('rag_prompt', ''),
                ', '.join(config.get('chat', {}).get('models', {}).get('available', [])),
                config.get('chat', {}).get('models', {}).get('default', ''),
                int(config.get('scraping', {}).get('chunk_size', 1000)),
                int(config.get('scraping', {}).get('overlap', 200)),
                config.get('pytorch_device', 'cpu'),
                "Settings refreshed"
            ]
        except (ValueError, TypeError) as e:
            # Log the specific value that failed conversion
            logger.error("Conversion error in refresh_settings():")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Current config values:")
            logger.error(f"rate_limit: {config.get('rate_limit')} ({type(config.get('rate_limit'))})")
            logger.error(f"retry_base_delay: {config.get('chat', {}).get('retry_base_delay')} ({type(config.get('chat', {}).get('retry_base_delay'))})")
            logger.error(f"retry_max_delay: {config.get('chat', {}).get('retry_max_delay')} ({type(config.get('chat', {}).get('retry_max_delay'))})")
            logger.error(f"pytorch_device: {config.get('pytorch_device')} ({type(config.get('pytorch_device'))})")
            return None

    with gr.Group():
        gr.Markdown("### Basic Settings")
        proxies_input = gr.Textbox(
            label="Proxies (comma-separated IPs)",
            value=', '.join(current_config.get('proxies', [])),
            lines=2
        )
        rate_limit_input = gr.Number(
            label="Rate Limit (requests per second)",
            value=current_config.get('rate_limit', 3),
            precision=0
        )
        user_agent_input = gr.Textbox(
            label="User Agent",
            value=current_config.get('user_agent', ''),
            lines=2
        )
        pytorch_device_input = gr.Textbox(
            label="PyTorch Device",
            value=current_config.get('pytorch_device', 'cpu'),
            info="Enter the PyTorch device (e.g., 'cuda' or 'cpu'). NOTE: A full application restart is required for changes to take effect."
        )
            
        with gr.Group():
            gr.Markdown("### Embeddings Settings")
            embedding_models_input = gr.Textbox(
                label="Available Embedding Models (comma-separated)",
                value=', '.join(current_config.get('embeddings', {}).get('models', {}).get('available', [])),
                lines=2
            )
            default_embedding_input = gr.Textbox(
                label="Default Embedding Model",
                value=current_config.get('embeddings', {}).get('models', {}).get('default', '')
            )
            reranker_model_input = gr.Textbox(
                label="Reranker Model",
                value=current_config.get('embeddings', {}).get('models', {}).get('reranker', '')
            )
            initial_percent_input = gr.Slider(
                label="Initial Retrieval Percentage",
                minimum=0.0,
                maximum=1.0,
                value=current_config.get('embeddings', {}).get('retrieval', {}).get('initial_percent', 0.2),
                step=0.1
            )
            min_initial_input = gr.Number(
                label="Minimum Initial Results",
                value=current_config.get('embeddings', {}).get('retrieval', {}).get('min_initial', 50),
                precision=0
            )
            max_initial_input = gr.Number(
                label="Maximum Initial Results",
                value=current_config.get('embeddings', {}).get('retrieval', {}).get('max_initial', 500),
                precision=0
            )
        
        with gr.Group():
            gr.Markdown("### Chunking Settings")
            chunk_size_input = gr.Number(
                label="Chunk Size",
                value=current_config.get('chunking', {}).get('chunk_size', 200),
                precision=0
            )
            max_chunk_size_input = gr.Number(
                label="Maximum Chunk Size",
                value=current_config.get('chunking', {}).get('max_chunk_size', 200),
                precision=0
            )
            chunk_overlap_input = gr.Number(
                label="Chunk Overlap",
                value=current_config.get('chunking', {}).get('chunk_overlap', 0),
                precision=0
            )
        
        with gr.Group():
            gr.Markdown("### Chat Settings")
            msg_history_size_input = gr.Number(
                label="Message History Size",
                value=current_config.get('chat', {}).get('message_history_size', 10),
                precision=0
            )
            max_retries_input = gr.Number(
                label="Maximum Retries",
                value=current_config.get('chat', {}).get('max_retries', 3),
                precision=0
            )
            retry_base_delay_input = gr.Number(
                label="Retry Base Delay (seconds)",
                value=current_config.get('chat', {}).get('retry_base_delay', 1),
                precision=0
            )
            retry_max_delay_input = gr.Number(
                label="Retry Maximum Delay (seconds)",
                value=current_config.get('chat', {}).get('retry_max_delay', 30),
                precision=0
            )
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                value=current_config.get('chat', {}).get('system_prompt', ''),
                lines=3
            )
            rag_prompt_input = gr.Textbox(
                label="RAG Prompt",
                value=current_config.get('chat', {}).get('rag_prompt', ''),
                lines=5
            )
            chat_models_input = gr.Textbox(
                label="Available Chat Models (comma-separated)",
                value=', '.join(current_config.get('chat', {}).get('models', {}).get('available', [])),
                lines=3
            )
            default_chat_model_input = gr.Textbox(
                label="Default Chat Model",
                value=current_config.get('chat', {}).get('models', {}).get('default', '')
            )
        
        with gr.Group():
            gr.Markdown("### Scraping Settings")
            scraping_chunk_size_input = gr.Number(
                label="Scraping Chunk Size",
                value=current_config.get('scraping', {}).get('chunk_size', 1000),
                precision=0
            )
            scraping_overlap_input = gr.Number(
                label="Scraping Overlap",
                value=current_config.get('scraping', {}).get('overlap', 200),
                precision=0
            )
        
        with gr.Group():
            gr.Markdown("### Actions")
            save_button = gr.Button("Save All Settings", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        # List of all components that need to be updated on refresh
        all_components = [
            proxies_input, rate_limit_input, user_agent_input,
            embedding_models_input, default_embedding_input, reranker_model_input,
            initial_percent_input, min_initial_input, max_initial_input,
            chunk_size_input, max_chunk_size_input, chunk_overlap_input,
            msg_history_size_input, max_retries_input, retry_base_delay_input, retry_max_delay_input,
            system_prompt_input, rag_prompt_input,
            chat_models_input, default_chat_model_input,
            scraping_chunk_size_input, scraping_overlap_input,
            pytorch_device_input,
            status_output
        ]
        
        # Connect the save button with automatic refresh
        save_button.click(
            fn=save_all_settings,
            inputs=[
                proxies_input, rate_limit_input, user_agent_input,
                embedding_models_input, default_embedding_input, reranker_model_input,
                initial_percent_input, min_initial_input, max_initial_input,
                chunk_size_input, max_chunk_size_input, chunk_overlap_input,
                msg_history_size_input, max_retries_input, retry_base_delay_input, retry_max_delay_input,
                system_prompt_input, rag_prompt_input,
                chat_models_input, default_chat_model_input,
                scraping_chunk_size_input, scraping_overlap_input,
                pytorch_device_input
            ],
            outputs=status_output
        ).then(
            fn=refresh_settings,
            inputs=[],
            outputs=all_components
        )
        
        return save_button, status_output
