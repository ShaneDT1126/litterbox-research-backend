import requests
import argparse
import json
import time
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger

logger = get_logger(__name__)


def test_copilot_integration(api_url: str, student_id: str = "test_student"):
    """Test the LitterBox API endpoints for Copilot Studio integration"""
    logger.info(f"Testing API at {api_url} with student ID {student_id}")

    # Test health endpoint
    logger.info("Testing health endpoint...")
    health_response = requests.get(f"{api_url}/api/health")
    logger.info(f"Health response: {health_response.status_code} {health_response.json()}")

    # Test query endpoint with a new conversation
    logger.info("Testing query endpoint with a new conversation...")
    query_data = {
        "query": "Hello, my name is Test User",
        "conversation_id": "new",
        "student_id": student_id
    }

    query_response = requests.post(f"{api_url}/api/query", json=query_data)
    logger.info(f"Query response: {query_response.status_code}")

    if query_response.status_code == 200:
        query_result = query_response.json()
        conversation_id = query_result["conversation_id"]
        logger.info(f"Conversation ID: {conversation_id}")
        logger.info(f"Response: {query_result['response']}")

        # Test query endpoint with an existing conversation
        logger.info("Testing query endpoint with an existing conversation...")
        query_data = {
            "query": "Can you tell me about CPU architecture?",
            "conversation_id": conversation_id,
            "student_id": student_id
        }

        query_response = requests.post(f"{api_url}/api/query", json=query_data)
        logger.info(f"Query response: {query_response.status_code}")

        if query_response.status_code == 200:
            query_result = query_response.json()
            logger.info(f"Response: {query_result['response']}")
            logger.info(f"Topic: {query_result['topic']}")
            logger.info(f"Scaffolding level: {query_result['scaffolding_level']}")

            # Test a follow-up question
            logger.info("Testing follow-up question...")
            query_data = {
                "query": "I understand that. Can you tell me more about the ALU?",
                "conversation_id": conversation_id,
                "student_id": student_id
            }

            query_response = requests.post(f"{api_url}/api/query", json=query_data)
            logger.info(f"Query response: {query_response.status_code}")

            if query_response.status_code == 200:
                query_result = query_response.json()
                logger.info(f"Response: {query_result['response']}")
                logger.info(f"Topic: {query_result['topic']}")
                logger.info(f"Scaffolding level: {query_result['scaffolding_level']}")

    logger.info("API testing complete!")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the LitterBox API for Copilot Studio integration')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                        help='URL of the LitterBox API')
    parser.add_argument('--student_id', type=str, default='test_student',
                        help='Student ID to use for testing')

    args = parser.parse_args()

    # Test the API
    test_copilot_integration(args.api_url, args.student_id)


if __name__ == "__main__":
    main()