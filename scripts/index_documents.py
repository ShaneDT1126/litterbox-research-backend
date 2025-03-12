import os
import argparse
import sys
import glob
from typing import List, Dict, Any
import json
import re
from langchain_community.vectorstores.utils import filter_complex_metadata

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.vector_store import ChromaVectorStore

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def extract_metadata(text: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Extract metadata from document text using LLM"""
    prompt = f"""
    You are an expert in Computer Organization Architecture. 
    Read the following text and extract metadata about it.

    Text: {text[:2000]}...

    Please provide the following metadata in JSON format:
    1. topic: The main topic from this list [CPU Architecture, Cache Memory, Memory Systems, Instruction Set Architecture, Pipelining, I/O Systems, Performance, General Computer Architecture]
    2. subtopic: A more specific subtopic within the main topic (2-5 words)
    3. difficulty: A rating from 1-5 where 1 is introductory and 5 is advanced
    4. keywords: A list of 5-10 important keywords from the text

    Your response must be valid JSON only, with no other text. Format it like this:
    {{
      "topic": "CPU Architecture",
      "subtopic": "ALU Design",
      "difficulty": 3,
      "keywords": ["processor", "arithmetic", "logic", "registers", "control unit"]
    }}
    """

    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # Debug: Print the actual response
    logger.debug(f"LLM response: {response_text}")

    try:
        # Try to parse as JSON directly
        metadata = json.loads(response_text)
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata JSON: {e}")

        # Try to extract JSON from the response if it contains other text
        try:
            # Look for JSON-like content between curly braces
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                metadata = json.loads(json_str)
                logger.info("Successfully extracted JSON from response")
                return metadata
        except Exception as e2:
            logger.error(f"Failed to extract JSON from response: {e2}")

        # Try to extract topic directly if JSON parsing fails
        topic = "General Computer Architecture"
        subtopic = "Unknown"
        difficulty = 3
        keywords = []

        # Try to extract topic from response
        topic_match = re.search(r'topic["\s:]+([^",\n]+)', response_text, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip()

        # Try to extract subtopic from response
        subtopic_match = re.search(r'subtopic["\s:]+([^",\n]+)', response_text, re.IGNORECASE)
        if subtopic_match:
            subtopic = subtopic_match.group(1).strip()

        # Try to extract difficulty from response
        difficulty_match = re.search(r'difficulty["\s:]+(\d)', response_text, re.IGNORECASE)
        if difficulty_match:
            difficulty = int(difficulty_match.group(1))

        # Try to extract keywords from response
        keywords_match = re.search(r'keywords["\s:]+\[(.*?)\]', response_text, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            keywords_str = keywords_match.group(1)
            keywords = [k.strip().strip('"\'') for k in keywords_str.split(',')]

        # Return default metadata
        return {
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "keywords": keywords
        }


def index_documents(content_dir: str, db_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Index documents from a directory into ChromaDB"""
    logger.info(f"Indexing documents from {content_dir} into {db_dir}")

    # Initialize LLM
    llm = ChatOpenAI(temperature=0.0, model=settings.LLM_MODEL)

    # Initialize ChromaDB vector store
    vector_store = ChromaVectorStore(persist_directory=db_dir)

    # Load documents with error handling
    logger.info("Loading documents...")
    documents = []

    # Get all PDF files in the directory
    pdf_files = glob.glob(os.path.join(content_dir, "**/*.pdf"), recursive=True)

    if not pdf_files:
        logger.warning(f"No PDF files found in {content_dir}")
        # Check for text files as fallback
        text_files = glob.glob(os.path.join(content_dir, "**/*.txt"), recursive=True)
        if text_files:
            logger.info(f"Found {len(text_files)} text files instead")
            for file_path in text_files:
                try:
                    logger.info(f"Loading {file_path}...")
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error loading file {file_path}")
                    logger.error(str(e))
                    continue
    else:
        logger.info(f"Found {len(pdf_files)} PDF files")
        for file_path in pdf_files:
            try:
                logger.info(f"Loading {file_path}...")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading file {file_path}")
                logger.error(str(e))
                continue

    logger.info(f"Successfully loaded {len(documents)} documents")

    if not documents:
        logger.error("No documents were successfully loaded. Exiting.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    logger.info("Splitting documents into chunks...")
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks")

    # Process each chunk
    processed_docs = []
    for i, doc in enumerate(split_docs):
        logger.info(f"Processing chunk {i + 1}/{len(split_docs)}...")

        # Extract metadata
        metadata = extract_metadata(doc.page_content, llm)

        # Add original metadata
        metadata.update(doc.metadata)

        # Filter complex metadata for ChromaDB compatibility
        filtered_metadata = filter_complex_metadata(metadata)

        # Add to processed docs
        processed_docs.append({
            "content": doc.page_content,
            "metadata": filtered_metadata
        })

    # Add to vector store
    logger.info("Adding documents to vector store...")
    vector_store.add_documents(processed_docs)
    logger.info(f"Added {len(processed_docs)} documents to vector store")

    logger.info("Indexing complete!")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Index documents for LitterBox')
    parser.add_argument('--content_dir', type=str, default='data/content',
                        help='Directory containing documents to index')
    parser.add_argument('--db_dir', type=str, default=settings.CHROMA_PERSIST_DIRECTORY,
                        help='Directory to store the vector database')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of document chunks')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                        help='Overlap between document chunks')

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.content_dir, exist_ok=True)
    os.makedirs(args.db_dir, exist_ok=True)

    # Index documents
    index_documents(args.content_dir, args.db_dir, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()