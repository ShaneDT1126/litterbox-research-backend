import requests
import argparse
import json
import time
import sys
import os
import uuid

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger

logger = get_logger(__name__)


def test_api(api_url: str, student_id: str = None):
    """Test the LitterBox API endpoints"""
    # Generate a unique student ID if not provided
    if not student_id:
        student_id = f"test_student_{uuid.uuid4().hex[:8]}"

    logger.info(f"Testing API at {api_url} with student ID {student_id}")

    # Test health endpoint
    logger.info("\n=== Testing health endpoint ===")
    try:
        health_response = requests.get(f"{api_url}/api/health")
        logger.info(f"Status code: {health_response.status_code}")
        logger.info(f"Response: {health_response.json()}")
        assert health_response.status_code == 200, "Health endpoint failed"
        assert health_response.json()["status"] == "healthy", "Health status not healthy"
        logger.info("✅ Health endpoint test passed")
    except Exception as e:
        logger.error(f"❌ Health endpoint test failed: {str(e)}")
        return False

    # Test topics endpoint
    logger.info("\n=== Testing topics endpoint ===")
    try:
        topics_response = requests.get(f"{api_url}/api/topics")
        logger.info(f"Status code: {topics_response.status_code}")
        logger.info(f"Response: {topics_response.json()}")
        assert topics_response.status_code == 200, "Topics endpoint failed"
        assert "topics" in topics_response.json(), "Topics not in response"
        assert isinstance(topics_response.json()["topics"], list), "Topics not a list"
        logger.info("✅ Topics endpoint test passed")
    except Exception as e:
        logger.error(f"❌ Topics endpoint test failed: {str(e)}")
        return False

    # Test query endpoint with a new conversation
    logger.info("\n=== Testing query endpoint (new conversation) ===")
    try:
        query_data = {
            "query": "Hello, my name is Test User",
            "conversation_id": "new",
            "student_id": student_id
        }

        logger.info(f"Request data: {query_data}")
        query_response = requests.post(f"{api_url}/api/query", json=query_data)
        logger.info(f"Status code: {query_response.status_code}")

        assert query_response.status_code == 200, "Query endpoint failed"

        query_result = query_response.json()
        logger.info(f"Response: {query_result['response'][:100]}...")
        logger.info(f"Topic: {query_result['topic']}")
        logger.info(f"Scaffolding level: {query_result['scaffolding_level']}")
        logger.info(f"Conversation ID: {query_result['conversation_id']}")

        conversation_id = query_result["conversation_id"]
        assert conversation_id != "new", "Conversation ID not updated"
        logger.info("✅ Query endpoint (new conversation) test passed")
    except Exception as e:
        logger.error(f"❌ Query endpoint (new conversation) test failed: {str(e)}")
        return False

    # Test query endpoint with an existing conversation
    logger.info("\n=== Testing query endpoint (existing conversation) ===")
    try:
        query_data = {
            "query": "Can you tell me about CPU architecture?",
            "conversation_id": conversation_id,
            "student_id": student_id
        }

        logger.info(f"Request data: {query_data}")
        query_response = requests.post(f"{api_url}/api/query", json=query_data)
        logger.info(f"Status code: {query_response.status_code}")

        assert query_response.status_code == 200, "Query endpoint failed"

        query_result = query_response.json()
        logger.info(f"Response: {query_result['response'][:100]}...")
        logger.info(f"Topic: {query_result['topic']}")
        logger.info(f"Scaffolding level: {query_result['scaffolding_level']}")
        logger.info(f"Conversation ID: {query_result['conversation_id']}")

        assert query_result["conversation_id"] == conversation_id, "Conversation ID changed"
        logger.info("✅ Query endpoint (existing conversation) test passed")
    except Exception as e:
        logger.error(f"❌ Query endpoint (existing conversation) test failed: {str(e)}")
        return False

    # Test query endpoint with a follow-up question
    logger.info("\n=== Testing query endpoint (follow-up question) ===")
    try:
        query_data = {
            "query": "I understand that. Can you tell me more about the ALU?",
            "conversation_id": conversation_id,
            "student_id": student_id
        }

        logger.info(f"Request data: {query_data}")
        query_response = requests.post(f"{api_url}/api/query", json=query_data)
        logger.info(f"Status code: {query_response.status_code}")

        assert query_response.status_code == 200, "Query endpoint failed"

        query_result = query_response.json()
        logger.info(f"Response: {query_result['response'][:100]}...")
        logger.info(f"Topic: {query_result['topic']}")
        logger.info(f"Scaffolding level: {query_result['scaffolding_level']}")
        logger.info(f"Conversation ID: {query_result['conversation_id']}")

        assert query_result["conversation_id"] == conversation_id, "Conversation ID changed"
        logger.info("✅ Query endpoint (follow-up question) test passed")
    except Exception as e:
        logger.error(f"❌ Query endpoint (follow-up question) test failed: {str(e)}")
        return False

    logger.info("\n=== All API tests passed successfully! ===")
    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the LitterBox API')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                        help='URL of the LitterBox API')
    parser.add_argument('--student_id', type=str, default=None,
                        help='Student ID to use for testing')

    args = parser.parse_args()

    # Test the API
    success = test_api(args.api_url, args.student_id)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()