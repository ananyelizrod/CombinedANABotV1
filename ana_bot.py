#!/usr/bin/env python3
"""
A.N.A. Bot - Fast Startup Chat Interface
Loads pre-computed embeddings for instant search capability.

Prerequisites:
- Run pdf_converter.py --create-embeddings first
- Requires: embeddings.npy, metadata.json, index.faiss

Usage:
    python ana_bot.py
""" 
# To run the program, I recommend using the live shareable link as opposed to the local host, I haven't been able
# to figure out why but the live works all of the time and the local only works sometimes. 

import os
import faiss
import numpy as np
import json
import base64
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM


# Function to convert image to base64
# (for the FDOT logo for the header of the test website) 
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Logo file not found: {image_path}")
        return None


# Load logo (update this path to your actual logo file)
LOGO_PATH = r"C:\Users\anany\Downloads\logo.png"  # Change this to your actual logo filename
logo_base64 = image_to_base64(LOGO_PATH)

# Configuration
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
FAISS_INDEX_FILE = "index.faiss"
LLM_MODEL = "llama3.2:latest"


# Load index and metadata
def load_index():
    """Load pre-computed embeddings, metadata, and FAISS index."""
    missing_files = []

    if not os.path.exists(FAISS_INDEX_FILE):
        missing_files.append(FAISS_INDEX_FILE)
    if not os.path.exists(EMBEDDINGS_FILE):
        missing_files.append(EMBEDDINGS_FILE)
    if not os.path.exists(METADATA_FILE):
        missing_files.append(METADATA_FILE)

    if missing_files:
        print(" Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n To create these files:")
        print("   python pdf_converter.py --create-embeddings")
        raise FileNotFoundError("Required A.N.A. Bot files not found")

    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "r", encoding='utf-8') as f:
            documents = json.load(f)

        print(f" Loaded {len(documents)} document chunks")
        return index, documents

    except Exception as e:
        print(f" Error loading files: {e}")
        raise


# Initialize components
def initialize_system():
    """Initialize A.N.A. Bot system components."""
    try:
        print(" Starting A.N.A. Bot...")

        # Load pre-computed files
        index, documents = load_index()

        # Load embedding model (same as used in converter)
        print(" Loading embedding model...")
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)

        print(" A.N.A. Bot ready!")
        return index, documents, embedding_model

    except Exception as e:
        print(f" System initialization failed: {e}")
        return None, None, None


# Initialize system on startup
index, documents, embedding_model = initialize_system()


# Search logic
def query_documents(query, top_k=3):
    """Search for relevant document chunks."""
    if not index or not documents or not embedding_model:
        return [], []

    try:
        query_embedding = embedding_model.encode([query])
        D, I = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)

        retrieved_chunks = []
        retrieved_sources = []

        for i in I[0]:
            if i < len(documents):
                chunk = documents[i]
                retrieved_chunks.append(chunk["text"])
                source_info = f"{chunk['source']} (Page {chunk['page']})"
                retrieved_sources.append(source_info)

        return retrieved_chunks, retrieved_sources

    except Exception as e:
        print(f"Error in document search: {e}")
        return [], []


# AI response logic
def generate_ai_response(query, retrieved_chunks, retrieved_sources):
    """Generate AI response using retrieved context."""
    if not retrieved_chunks:
        return "No relevant information found in the documents.", ""

    try:
        llm = OllamaLLM(model=LLM_MODEL)
        context = "\n\n".join(retrieved_chunks)

        response = llm.invoke(
            f"Use the following document excerpts to answer the question:\n\n{context}\n\nQuestion: {query}"
        )

        # Format sources
        sources_text = "\n".join([f"• {source}" for source in retrieved_sources])
        return response, sources_text

    except Exception as e:
        return f"Error generating AI response: {str(e)}", ""


# Main search function for Gradio
def search_manuals(query):
    """Main search function called by Gradio interface."""
    if not query or not query.strip():
        return "Please enter a valid search query.", ""

    if not index or not documents or not embedding_model:
        return "System not properly initialized. Please run pdf_converter.py --create-embeddings first.", ""

    # Retrieve relevant document chunks
    retrieved_chunks, retrieved_sources = query_documents(query.strip())

    # Generate AI response
    ai_response, sources = generate_ai_response(query.strip(), retrieved_chunks, retrieved_sources)

    return ai_response, sources


# Create interface
def create_interface():
    """Create Gradio interface for A.N.A. Bot."""
    # Create custom dark theme
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Times New Roman"), "serif"]
    ).set(
        # Dark theme colors
        body_background_fill="#0f172a",
        body_text_color="#f8fafc",
        background_fill_primary="#1e293b",
        background_fill_secondary="#334155",
        border_color_primary="#475569",
        block_background_fill="#1e293b",
        block_border_color="#475569",
        block_label_text_color="#e2e8f0",
        input_background_fill="#334155",
        input_border_color="#64748b",
        input_placeholder_color="#94a3b8",
        button_primary_background_fill="#3b82f6",
        button_primary_text_color="#ffffff"
    )

    with gr.Blocks(theme=custom_theme, title="A.N.A. BOT", css="""
        .gr-examples .gr-button {
            background: #475569 !important;
            color: #e2e8f0 !important;
            border: 1px solid #64748b !important;
        }
        .gr-examples .gr-button:hover {
            background: #64748b !important;
            color: #f1f5f9 !important;
        }
    """) as interface:

        # Header with logo
        if logo_base64:
            gr.HTML(f"""
                <div style="display: flex; align-items: center; padding: 20px 0; border-bottom: 1px solid #475569; margin-bottom: 30px;">
                    <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 40px; margin-right: 15px;" />
                    <h1 style="margin: 0; color: #f8fafc; font-size: 1.5rem; font-weight: 600;">A.N.A. BOT</h1>
                </div>
            """)
        else:
            gr.HTML("""
                <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid #475569; margin-bottom: 30px;">
                    <h1 style="margin: 0; color: #f8fafc; font-size: 1.8rem; font-weight: 600;">A.N.A. BOT</h1>
                </div>
            """)

        # System status
        if index and documents and embedding_model:
            status_color = "#10b981"  # Green
            status_text = f" Ready - {len(documents)} manual chunks loaded"
        else:
            status_color = "#ef4444"  # Red
            status_text = " System not ready - Run pdf_converter.py --create-embeddings"

        gr.HTML(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <p style="color: {status_color}; font-size: 14px; font-weight: 500;">{status_text}</p>
                <p style="color: #cbd5e1; font-size: 16px;">Search your technical documentation with AI-powered answers</p>
            </div>
        """)

        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Ask me anything about your manuals...",
            lines=1
        )

        search_btn = gr.Button("Search", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                response_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
                gr.HTML("""
                    <p style="color: #94a3b8; font-size: 12px; margin-top: 5px; margin-bottom: 20px;">
                        A.N.A. Bot can make mistakes. Please verify important information.
                    </p>
                """)

        with gr.Row():
            sources_output = gr.Textbox(
                label="Sources",
                lines=4,
                interactive=False
            )

        # Event handlers
        search_btn.click(
            fn=search_manuals,
            inputs=[query_input],
            outputs=[response_output, sources_output]
        )

        query_input.submit(
            fn=search_manuals,
            inputs=[query_input],
            outputs=[response_output, sources_output]
        )

        # Example queries
        with gr.Accordion("Example Queries", open=False):
            gr.Examples(
                examples=[
                    ["What are the formulas for determining taper length"],
                    ["How do i warrant the need for a flashing beacon ?"],
                    ["What are the spacing requirements for guide signs on rural highways?"],
                    ["Under what conditions is a traffic signal retiming study required?"],
                    ["How do I warrant the need for a traffic signal?"]
                ],
                inputs=[query_input],
                outputs=[response_output, sources_output],
                fn=search_manuals,
                cache_examples=False
            )

    return interface


# Launch the application
if __name__ == "__main__":
    if not (index and documents and embedding_model):
        print("\n Setup Required:")
        print("   1. Add PDF files to 'manuals/' folder")
        print("   2. Run: python pdf_converter.py --create-embeddings")
        print("   3. Then run: python ana_bot.py")
        exit(1)

    print(f"\n Launching A.N.A. Bot interface...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True

    )
