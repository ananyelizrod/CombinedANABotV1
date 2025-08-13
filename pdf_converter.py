# meant for annual processing, a slow process
#can be used to update once yearly for maintenance

# !/usr/bin/env python3
"""
PDF Converter for A.N.A. Bot
Standalone program to process PDF manuals and generate embeddings.
Run this once per year when manuals are updated.

Usage:
    python pdf_converter.py                    # Convert PDFs to JSON
    python pdf_converter.py --create-embeddings # Full pipeline: PDF -> Embeddings
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not found. Install with: pip install PyMuPDF")
    exit(1)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("⚠️  LangChain not found. Using simple text splitting.")
    print("   For better chunking, install with: pip install langchain")
    RecursiveCharacterTextSplitter = None


class PDFConverter:
    """Convert PDF manuals to structured JSON and embeddings for A.N.A. Bot."""

    def __init__(self,
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200,
                 log_level: str = "INFO"):
        """
        Initialize the PDF converter.

        Args:
            chunk_size: Maximum size of text chunks (optimized for A.N.A. Bot)
            chunk_overlap: Overlap between consecutive chunks
            log_level: Logging level
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize text splitter optimized for technical manuals
        if RecursiveCharacterTextSplitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        else:
            self.text_splitter = None

    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata

            clean_metadata = {
                "title": metadata.get("title", "").strip() or Path(pdf_path).stem,
                "author": metadata.get("author", "").strip() or "Unknown",
                "subject": metadata.get("subject", "").strip(),
                "creator": metadata.get("creator", "").strip(),
                "created": metadata.get("creationDate", "").strip(),
                "pages": doc.page_count,
                "file_size": os.path.getsize(pdf_path)
            }

            doc.close()
            return clean_metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {"title": Path(pdf_path).stem, "author": "Unknown", "pages": 0}

    def clean_text(self, text: str) -> str:
        """Clean text while preserving technical manual formatting."""
        if not text:
            return ""

        # Preserve technical formatting
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)

        # Remove page headers/footers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE)

        # Keep technical symbols common in manuals
        text = re.sub(r'[^\w\s\.,;:!?\'"()\-\n\r°%#&/\\§\[\]{}]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)

        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []

        if self.text_splitter:
            return self.text_splitter.split_text(text)
        else:
            return self.simple_chunk_text(text)

    def simple_chunk_text(self, text: str) -> List[str]:
        """Simple chunking fallback."""
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1

            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) + 1 for w in current_chunk)

            current_chunk.append(word)
            current_size += word_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_chunk_id(self, filename: str, page_num: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        base_name = Path(filename).stem
        return f"{base_name}_p{page_num:03d}_c{chunk_index:03d}"

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file."""
        self.logger.info(f"Processing: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            pdf_metadata = self.extract_pdf_metadata(pdf_path)
            filename = os.path.basename(pdf_path)
            all_chunks = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                raw_text = page.get_text()

                if not raw_text.strip():
                    continue

                cleaned_text = self.clean_text(raw_text)
                if not cleaned_text:
                    continue

                chunks = self.chunk_text(cleaned_text)

                for chunk_index, chunk_text in enumerate(chunks):
                    if len(chunk_text.strip()) < 100:  # Skip short chunks
                        continue

                    # A.N.A. Bot compatible format
                    chunk_data = {
                        "text": chunk_text.strip(),
                        "source": filename,
                        "page": page_num + 1,
                        "id": self.generate_chunk_id(filename, page_num + 1, chunk_index),
                        "chunk_index": chunk_index,
                        "metadata": {
                            "source_path": pdf_path,
                            "text_length": len(chunk_text.strip()),
                            "pdf_metadata": pdf_metadata,
                            "extraction_timestamp": datetime.now().isoformat(),
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap
                        }
                    }

                    all_chunks.append(chunk_data)

            doc.close()
            self.logger.info(f"Extracted {len(all_chunks)} chunks from {filename}")
            return all_chunks

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return []

    def process_manuals_directory(self, manuals_dir: str = "manuals") -> List[Dict[str, Any]]:
        """Process all PDFs in the manuals directory."""
        manuals_path = Path(manuals_dir)

        if not manuals_path.exists():
            manuals_path.mkdir(exist_ok=True)
            self.logger.warning(f"Created {manuals_dir} directory. Please add PDF files.")
            return []

        pdf_files = list(manuals_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {manuals_dir}")
            return []

        self.logger.info(f"Found {len(pdf_files)} PDF files")
        all_chunks = []

        for pdf_file in pdf_files:
            chunks = self.process_pdf(str(pdf_file))
            all_chunks.extend(chunks)

        return all_chunks

    def save_json(self, chunks: List[Dict[str, Any]], output_dir: str = "json_output") -> str:
        """Save chunks to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"manual_chunks_{timestamp}.json"
        json_file = output_path / filename

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved {len(chunks)} chunks to: {json_file}")
            return str(json_file)

        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
            return ""

    def create_ana_bot_files(self, chunks: List[Dict[str, Any]],
                             model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
        """Create embeddings and files for A.N.A. Bot."""

        # Import embedding libraries
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            self.logger.error(f"Missing libraries for embeddings: {e}")
            self.logger.error("Install with: pip install faiss-cpu sentence-transformers torch")
            return False

        if not chunks:
            self.logger.error("No chunks to process")
            return False

        self.logger.info(f"Creating embeddings for {len(chunks)} chunks...")

        try:
            # Initialize embedding model
            embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

            # Extract texts
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings
            self.logger.info("Generating embeddings... This may take a while.")
            embeddings = embedding_model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Save files for A.N.A. Bot
            np.save("embeddings.npy", embeddings)
            self.logger.info("✅ Saved embeddings.npy")

            with open("metadata.json", 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.logger.info("✅ Saved metadata.json")

            faiss.write_index(index, "index.faiss")
            self.logger.info("✅ Saved index.faiss")

            self.logger.info(f"🎉 A.N.A. Bot files ready! ({len(chunks)} chunks, {dimension}D embeddings)")
            return True

        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return False

    def get_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing statistics."""
        if not chunks:
            return {"error": "No chunks to analyze"}

        total_text_length = sum(len(chunk["text"]) for chunk in chunks)
        sources = set(chunk["source"] for chunk in chunks)
        pages = set(chunk["page"] for chunk in chunks)

        return {
            "total_chunks": len(chunks),
            "total_text_length": total_text_length,
            "average_chunk_length": round(total_text_length / len(chunks), 2),
            "unique_sources": len(sources),
            "total_pages": len(pages),
            "sources": list(sources)
        }


def main():
    """Main function for PDF converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF Converter for A.N.A. Bot - Process manuals once per year",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_converter.py                       # Convert PDFs to JSON
  python pdf_converter.py --create-embeddings  # Full pipeline for A.N.A. Bot
  python pdf_converter.py --stats               # Show processing statistics
        """
    )

    parser.add_argument("--manuals-dir", default="manuals",
                        help="Directory with PDF manuals (default: manuals)")
    parser.add_argument("--create-embeddings", action="store_true",
                        help="Create embeddings and FAISS index for A.N.A. Bot")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Text chunk size (default: 2000)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Chunk overlap (default: 200)")
    parser.add_argument("--stats", action="store_true",
                        help="Show processing statistics")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Check for manuals directory
    if not Path(args.manuals_dir).exists():
        print(f"📁 Creating '{args.manuals_dir}' directory...")
        Path(args.manuals_dir).mkdir(exist_ok=True)
        print(f"   Please add PDF files to '{args.manuals_dir}' and run again.")
        return

    pdf_files = list(Path(args.manuals_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"📁 No PDF files found in '{args.manuals_dir}'")
        print(f"   Please add PDF files and run again.")
        return

    print(f"📚 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   📄 {pdf.name}")

    # Initialize converter
    converter = PDFConverter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        log_level=args.log_level
    )

    # Process PDFs
    print(f"\n🔄 Processing PDFs...")
    chunks = converter.process_manuals_directory(args.manuals_dir)

    if not chunks:
        print("❌ No chunks were extracted from PDFs")
        return

    # Save JSON (always)
    json_file = converter.save_json(chunks)
    if json_file:
        print(f"✅ JSON saved: {json_file}")

    # Create A.N.A. Bot files if requested
    if args.create_embeddings:
        print(f"\n🧠 Creating embeddings for A.N.A. Bot...")
        success = converter.create_ana_bot_files(chunks, args.embedding_model)

        if success:
            print(f"\n🎉 A.N.A. Bot is ready!")
            print(f"   📄 Files created: embeddings.npy, metadata.json, index.faiss")
            print(f"   🚀 Start your A.N.A. Bot: python ana_bot.py")
        else:
            print(f"\n❌ Failed to create embeddings")
            return
    else:
        print(f"\n💡 To create embeddings for A.N.A. Bot:")
        print(f"   python pdf_converter.py --create-embeddings")

    # Show statistics
    if args.stats:
        print(f"\n📊 Processing Statistics:")
        stats = converter.get_statistics(chunks)
        for key, value in stats.items():
            if key != "sources":
                print(f"   {key.replace('_', ' ').title()}: {value}")

    print(f"\n✨ PDF conversion complete!")


if __name__ == "__main__":
    main()