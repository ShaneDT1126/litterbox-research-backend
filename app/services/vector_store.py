from typing import List, Dict, Any, Optional
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Service for storing and retrieving vector embeddings"""

    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB vector store"""
        # Initialize OpenAI embeddings
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.embeddings = OpenAIEmbeddings()

        # Set persist directory
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Create or load the vector store
        if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            logger.info(f"Loading existing vector store from {persist_directory}")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.info(f"Creating new vector store at {persist_directory}")
            # Initialize with empty collection
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            # Note: persist() is no longer needed as ChromaDB automatically persists when persist_directory is provided

    def add_documents(self, documents: List[Dict[str, Any]], batch_size=100) -> int:
        """Add documents to the vector store in batches"""
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Convert batch to langchain Document format
            langchain_docs = []
            for doc in batch:
                # Process metadata to ensure it only contains simple types
                filtered_metadata = {}
                for key, value in doc["metadata"].items():
                    if isinstance(value, list):
                        # Convert lists to comma-separated strings
                        filtered_metadata[key] = ", ".join(str(item) for item in value)
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        # Keep simple types as they are
                        filtered_metadata[key] = value
                    else:
                        # Convert other complex types to strings
                        filtered_metadata[key] = str(value)

                langchain_docs.append(
                    Document(
                        page_content=doc["content"],
                        metadata=filtered_metadata
                    )
                )

            # Add batch to vector store
            self.vector_store.add_documents(langchain_docs)
            total_added += len(langchain_docs)

            # Log progress for large batches
            if len(documents) > batch_size:
                logger.info(
                    f"Added batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size} to vector store")

        # Note: persist() is no longer needed as ChromaDB automatically persists when persist_directory is provided

        logger.info(f"Added {total_added} documents to vector store")
        return total_added

    def search(self, query: str, filter_dict: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Document]:
        """Search for documents similar to the query"""
        if filter_dict:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )

        logger.info(f"Found {len(results)} documents for query: {query}")
        return results

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever for the vector store"""
        if search_kwargs is None:
            search_kwargs = {"k": settings.RETRIEVER_K}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def index_pdf_directory(self, directory_path: str) -> int:
        """Index all PDFs in a directory"""
        # Get all PDF files in the directory
        pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                     if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return 0

        # Process each PDF file
        all_docs = []
        for pdf_file in pdf_files:
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file
                    # Extract filename without extension as title if not present
                    if "title" not in doc.metadata:
                        doc.metadata["title"] = os.path.basename(pdf_file).replace('.pdf', '')

                all_docs.extend(documents)
                logger.info(f"Processed {pdf_file}: {len(documents)} pages")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        split_docs = text_splitter.split_documents(all_docs)
        logger.info(f"Split {len(all_docs)} documents into {len(split_docs)} chunks")

        # Process metadata for each document
        processed_docs = []
        for doc in split_docs:
            # Process metadata to ensure it only contains simple types
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    filtered_metadata[key] = ", ".join(str(item) for item in value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    # Keep simple types as they are
                    filtered_metadata[key] = value
                else:
                    # Convert other complex types to strings
                    filtered_metadata[key] = str(value)

            processed_docs.append({
                "content": doc.page_content,
                "metadata": filtered_metadata
            })

        # Add to vector store
        return self.add_documents(processed_docs)