import pytest
import os
import time
from app.services.conversation import ConversationMemory

# Skip tests if MongoDB connection string is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("MONGODB_CONNECTION_STRING"),
    reason="MongoDB connection string not set"
)


@pytest.fixture
def conversation_memory():
    """Create a ConversationMemory instance for testing"""
    return ConversationMemory()


def test_create_conversation(conversation_memory):
    """Test creating a new conversation"""
    conversation_id, exchanges = conversation_memory.get_conversation("new", "test_student")

    assert conversation_id is not None
    assert len(exchanges) == 0


def test_add_exchange(conversation_memory):
    """Test adding an exchange to a conversation"""
    conversation_id, _ = conversation_memory.get_conversation("new", "test_student")

    conversation_memory.add_exchange(
        conversation_id=conversation_id,
        student_message="Hello",
        assistant_message="Hi there!",
        topic="Introduction",
        scaffolding_level=2,
        student_id="test_student"
    )

    _, exchanges = conversation_memory.get_conversation(conversation_id)

    assert len(exchanges) == 1
    assert exchanges[0]["student_message"] == "Hello"
    assert exchanges[0]["assistant_message"] == "Hi there!"
    assert exchanges[0]["topic"] == "Introduction"
    assert exchanges[0]["scaffolding_level"] == 2