gradio_css = """
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
            --accent-color: transparent !important;
            outline: none !important;
        }
        
        /* Override Gradio's loading animation only for scraping progress */
        .scraping-progress.generating,
        .scraping-progress .generating {
            border: none !important;
            box-shadow: none !important;
            --accent-color: transparent !important;
        }
        
        .scraping-progress.progress,
        .scraping-progress .progress {
            border: none !important;
            box-shadow: none !important;
            --accent-color: transparent !important;
        }
        
        /* Ensure no animation effects on progress container children */
        .scraping-progress > *:not(.tqdm-status) {
            --accent-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        .scraping-progress:focus-within {
            outline: none !important;
            box-shadow: none !important;
            --accent-color: transparent !important;
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
        /* Make the parent row force both columns to stretch vertically. */
#outer-row {
  align-items: stretch;  /* Ensures columns match height if possible */
}

/* Give each column a vertical layout without added gap/padding. */
#left-col, #right-col {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;         /* You can adjust or remove the gap if you like */
  margin: 0; 
  padding: 0;
}

/* Force a minimum height so that the dropdown and textbox match visually.
   Adjust this to whatever height you want. */
#left-col .gr-dropdown,
#right-col .gr-textbox {
  min-height: 80px; 
}



/* If the checkboxes appear to cause extra row height,
   you can override their container styles, too. */
#right-col .gr-checkbox .wrap {
  margin: 0;
  padding: 0;
}

/* Example: unify the row alignment so the button + checkbox line up nicely. */
#right-col .gr-row {
  align-items: center; 
  gap: 1rem;   /* Adjust horizontal spacing between button & checkbox */
}

#status-text {
padding-top: 10.667px !important;
padding-bottom: 10.667px !important;
}
    """
