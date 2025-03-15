from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Service for storing and retrieving vector embeddings with Azure Blob Storage support"""

    def __init__(self, persist_directory: Optional[str] = None, use_azure: bool = True):
        """Initialize ChromaDB vector store with optional Azure Blob Storage"""
        # Initialize OpenAI embeddings
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.embeddings = OpenAIEmbeddings()

        # Set persist directory
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        self.persist_directory = persist_directory

        # Azure Blob Storage configuration
        self.use_azure = use_azure and settings.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME or "litterbox-vectorstore"
        self.blob_prefix = settings.AZURE_STORAGE_BLOB_PREFIX or "chroma-db"

        # Create local directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize Azure Blob Storage if enabled
        if self.use_azure:
            self._init_azure_storage()

        # Create or load the vector store
        self._init_vector_store()

    def _init_azure_storage(self):
        """Initialize Azure Blob Storage connection and container"""
        try:
            # Create blob service client
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )

            # Create container if it doesn't exist
            try:
                self.container_client = self.blob_service_client.create_container(self.container_name)
                logger.info(f"Created Azure Blob Storage container: {self.container_name}")
            except ResourceExistsError:
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                logger.info(f"Using existing Azure Blob Storage container: {self.container_name}")

            # Download existing database if it exists
            self._download_from_azure()

        except Exception as e:
            logger.error(f"Error initializing Azure Blob Storage: {e}")
            self.use_azure = False
            logger.warning("Falling back to local storage only")

    def _download_from_azure(self):
        """Download Chroma database files from Azure Blob Storage"""
        # List all blobs with our prefix
        blobs = list(self.container_client.list_blobs(name_starts_with=f"{self.blob_prefix}/"))

        if not blobs:
            logger.info(f"No existing database found in Azure Blob Storage with prefix {self.blob_prefix}")
            return

        logger.info(f"Downloading {len(blobs)} files from Azure Blob Storage")

        # Download each blob to the local persist directory
        for blob in blobs:
            # Get relative path from blob prefix
            relative_path = blob.name[len(f"{self.blob_prefix}/"):]
            local_path = os.path.join(self.persist_directory, relative_path)

            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download blob
            blob_client = self.container_client.get_blob_client(blob.name)
            with open(local_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

        logger.info(f"Downloaded Chroma database from Azure Blob Storage to {self.persist_directory}")

    def _upload_to_azure(self):
        """Upload Chroma database files to Azure Blob Storage"""
        if not self.use_azure:
            return

        try:
            # Walk through the persist directory and upload all files
            for root, _, files in os.walk(self.persist_directory):
                for file in files:
                    # Get local file path
                    local_path = os.path.join(root, file)

                    # Get relative path from persist directory
                    relative_path = os.path.relpath(local_path, self.persist_directory)

                    # Create blob path
                    blob_path = f"{self.blob_prefix}/{relative_path}"

                    # Upload file
                    blob_client = self.container_client.get_blob_client(blob_path)
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)

            logger.info(f"Uploaded Chroma database to Azure Blob Storage with prefix {self.blob_prefix}")
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob Storage: {e}")

    def _init_vector_store(self):
        """Initialize or load the Chroma vector store"""
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.info(f"Creating new vector store at {self.persist_directory}")
            # Initialize with empty collection
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

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

        # Upload to Azure Blob Storage after adding documents
        if self.use_azure:
            self._upload_to_azure()

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