import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from main import app
from app.services.rag_engine import RAGEngine

# Create test client
client = TestClient(app)

# Mock RAG engine response
mock_rag_response = {
    "response": "This is a test response",
    "sources": [
        {
            "content": "Test source content",
            "topic": "CPU Architecture",
            "confidence": 0.8,
            "metadata": {"source": "test_document.pdf"}
        }
    ],
    "scaffolding_level": 2,
    "topic": "CPU Architecture",
    "conversation_id": "test_conversation_123"
}


def test_health_endpoint():
    """Test the health endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_topics_endpoint():
    """Test the topics endpoint"""
    response = client.get("/api/topics")
    assert response.status_code == 200
    assert "topics" in response.json()
    assert isinstance(response.json()["topics"], list)


@patch.object(RAGEngine, 'process_query')
def test_query_endpoint(mock_process_query):
    """Test the query endpoint"""
    # Configure the mock
    mock_process_query.return_value = mock_rag_response

    # Test data
    test_data = {
        "query": "What is CPU architecture?",
        "conversation_id": "new",
        "student_id": "test_student"
    }

    # Make request
    response = client.post("/api/query", json=test_data)

    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["response"] == "This is a test response"
    assert result["topic"] == "CPU Architecture"
    assert result["scaffolding_level"] == 2
    assert result["conversation_id"] == "test_conversation_123"
    assert len(result["sources"]) == 1

    # Verify mock was called with correct arguments
    mock_process_query.assert_called_once_with(
        query="What is CPU architecture?",
        conversation_id="new",
        student_id="test_student"
    )


@patch.object(RAGEngine, 'process_query')
def test_query_endpoint_error_handling(mock_process_query):
    """Test error handling in the query endpoint"""
    # Configure the mock to raise an exception
    mock_process_query.side_effect = Exception("Test error")

    # Test data
    test_data = {
        "query": "What is CPU architecture?",
        "conversation_id": "new",
        "student_id": "test_student"
    }

    # Make request
    response = client.post("/api/query", json=test_data)

    # Check response
    assert response.status_code == 500
    assert "detail" in response.json()