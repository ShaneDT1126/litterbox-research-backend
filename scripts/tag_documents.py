import os
import argparse
import sys
import glob

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import ChromaVectorStore
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def extract_topic(text, llm):
    """Extract the topic from document text using LLM"""
    prompt = f"""
    You are an expert in Computer Organization Architecture. 
    Read the following text and determine which of these topics it primarily discusses:

    1. CPU Architecture
    2. Cache Memory
    3. Memory Systems
    4. Instruction Set Architecture
    5. Pipelining
    6. I/O Systems
    7. Performance
    8. General Computer Architecture (if it doesn't fit any specific category)

    Text: {text[:1000]}...

    Respond with ONLY the topic name, nothing else.
    """

    response = llm.invoke(prompt)
    return response.content.strip()


def extract_subtopic(text, topic, llm):
    """Extract a more specific subtopic based on the main topic"""
    prompt = f"""
    You are an expert in Computer Organization Architecture. 
    The following text is about {topic}.

    Read the text and determine a specific subtopic within {topic} that it discusses.
    Be concise but descriptive (2-5 words).

    Text: {text[:1000]}...

    Respond with ONLY the subtopic name, nothing else.
    """

    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tag documents for LitterBox chatbot')
    parser.add_argument('--content_dir', type=str, default='data/content',
                        help='Directory containing documents to tag')
    parser.add_argument('--db_dir', type=str, default=settings.CHROMA_PERSIST_DIRECTORY,
                        help='Directory to store ChromaDB')

    args = parser.parse_args()

    # Initialize LLM
    llm = ChatOpenAI(temperature=0.0, model=settings.LLM_MODEL)

    # Initialize ChromaDB vector store
    vector_store = ChromaVectorStore(persist_directory=args.db_dir)

    # Get all files in the directory
    pdf_files = glob.glob(os.path.join(args.content_dir, "**/*.pdf"), recursive=True)
    text_files = glob.glob(os.path.join(args.content_dir, "**/*.txt"), recursive=True)

    all_files = pdf_files + text_files

    if not all_files:
        logger.error(f"No PDF or text files found in {args.content_dir}")
        return

    # Process each file
    for file_path in all_files:
        try:
            logger.info(f"Processing {file_path}...")

            # Load document based on file type
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")

            # Process each chunk
            tagged_docs = []
            for i, doc in enumerate(split_docs):
                logger.info(f"Tagging chunk {i + 1}/{len(split_docs)}...")

                # Extract topic and subtopic
                topic = extract_topic(doc.page_content, llm)
                subtopic = extract_subtopic(doc.page_content, topic, llm)

                # Add metadata
                doc.metadata["topic"] = topic
                doc.metadata["subtopic"] = subtopic
                doc.metadata["source"] = file_path

                # Add to tagged docs
                tagged_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            # Add to vector store
            vector_store.add_documents(tagged_docs)
            logger.info(f"Added {len(tagged_docs)} tagged documents to ChromaDB")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info("Document tagging complete!")


if __name__ == "__main__":
    main()