import os
import argparse
import sys
import glob
from typing import List, Dict, Any
from tqdm import tqdm

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import ChromaVectorStore
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def fast_index_documents(content_dir: str, db_dir: str, chunk_size: int = 2000, chunk_overlap: int = 100,
                         max_files: int = None):
    """Fast index documents from a directory into ChromaDB"""
    logger.info(f"Fast indexing documents from {content_dir} into {db_dir}")

    # Initialize ChromaDB vector store
    vector_store = ChromaVectorStore(persist_directory=db_dir)

    # Load documents with error handling
    logger.info("Loading documents...")

    # Get all files in the directory
    pdf_files = glob.glob(os.path.join(content_dir, "**/*.pdf"), recursive=True)
    text_files = glob.glob(os.path.join(content_dir, "**/*.txt"), recursive=True)

    all_files = pdf_files + text_files

    if not all_files:
        logger.warning(f"No files found in {content_dir}")
        return

    # Limit number of files if specified
    if max_files and max_files > 0:
        all_files = all_files[:max_files]
        logger.info(f"Limited to processing {max_files} files")

    logger.info(f"Found {len(all_files)} files to process")

    # Process files in batches to avoid memory issues
    batch_size = 1  # Process one file at a time
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        batch_docs = []

        for file_path in batch_files:
            try:
                logger.info(f"Loading {file_path}...")
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                try:
                    file_docs = loader.load()
                    batch_docs.extend(file_docs)
                    logger.info(f"Loaded {len(file_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading content from {file_path}: {e}")
                    continue
            except Exception as e:
                logger.error(f"Error initializing loader for {file_path}: {e}")
                continue

        if not batch_docs:
            logger.warning(f"No documents loaded from batch {i // batch_size + 1}")
            continue

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        logger.info("Splitting documents into chunks...")
        split_docs = text_splitter.split_documents(batch_docs)
        logger.info(f"Split into {len(split_docs)} chunks")

        # Process chunks in batches
        chunk_batch_size = 100  # Process 100 chunks at a time
        for j in range(0, len(split_docs), chunk_batch_size):
            chunk_batch = split_docs[j:j + chunk_batch_size]
            logger.info(
                f"Processing chunk batch {j // chunk_batch_size + 1}/{(len(split_docs) + chunk_batch_size - 1) // chunk_batch_size}...")

            processed_batch = []
            for doc in tqdm(chunk_batch, desc="Processing chunks"):
                # Extract filename from path
                filename = os.path.basename(doc.metadata.get("source", "Unknown"))

                # Use simple metadata
                metadata = {
                    "topic": "Computer Architecture",
                    "subtopic": "General Concepts",
                    "difficulty": 3,
                    "keywords": "computer architecture, systems, design",
                    "source": filename,
                    "page": doc.metadata.get("page", 0)
                }

                # Add to processed docs
                processed_batch.append({
                    "content": doc.page_content,
                    "metadata": metadata
                })

            # Add batch to vector store
            vector_store.add_documents(processed_batch)
            logger.info(f"Added batch of {len(processed_batch)} chunks to vector store")

    logger.info("Fast indexing complete!")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fast index documents for LitterBox')
    parser.add_argument('--content_dir', type=str, default='data/content',
                        help='Directory containing documents to index')
    parser.add_argument('--db_dir', type=str, default=settings.CHROMA_PERSIST_DIRECTORY,
                        help='Directory to store the vector database')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Size of document chunks')
    parser.add_argument('--chunk_overlap', type=int, default=100,
                        help='Overlap between document chunks')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.content_dir, exist_ok=True)
    os.makedirs(args.db_dir, exist_ok=True)

    # Index documents
    fast_index_documents(args.content_dir, args.db_dir, args.chunk_size, args.chunk_overlap, args.max_files)


if __name__ == "__main__":
    main()