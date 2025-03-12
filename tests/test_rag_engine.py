import pytest
import os
from app.services.rag_engine import RAGEngine

# Skip tests if OpenAI API key is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OpenAI API key not set"
)

@pytest.fixture
def rag_engine():
    """Create a RAGEngine instance for testing"""
    return RAGEngine()

def test_is_introduction(rag_engine):
    """Test the is_introduction method"""
    assert rag_engine.is_introduction("Hello, my name is John")
    assert rag_engine.is_introduction("Hi, I am Jane")
    assert not rag_engine.is_introduction("What is CPU architecture?")

def test_detect_topic(rag_engine):
    """Test the detect_topic method"""
    assert rag_engine.detect_topic("What is a CPU?") == "CPU Architecture"
    assert rag_engine.detect_topic("How does cache memory work?") == "Cache Memory"
    assert rag_engine.detect_topic("Hello, my name is John") == "Introduction"