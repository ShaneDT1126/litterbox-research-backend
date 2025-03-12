import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # MongoDB Configuration
    MONGODB_CONNECTION_STRING: str = os.getenv("MONGODB_CONNECTION_STRING", "")
    MONGODB_DATABASE: str = "litterbox"
    MONGODB_COLLECTION: str = "conversations"

    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")

    # RAG Configuration
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Scaffolding Configuration
    DEFAULT_SCAFFOLDING_LEVEL: int = int(os.getenv("DEFAULT_SCAFFOLDING_LEVEL", "2"))

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create a global settings object
settings = Settings()