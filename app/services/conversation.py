from typing import List, Dict, Tuple, Optional
import pymongo
import time
import uuid
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Service for storing and retrieving conversation history"""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize MongoDB connection for conversation memory"""
        if connection_string is None:
            connection_string = settings.MONGODB_CONNECTION_STRING
            if not connection_string:
                raise ValueError("MONGODB_CONNECTION_STRING environment variable is not set")

        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[settings.MONGODB_DATABASE]
        self.conversations = self.db[settings.MONGODB_COLLECTION]

        # Create indexes for faster queries
        self.conversations.create_index("conversation_id")
        self.conversations.create_index("student_id")
        logger.info("Connected to MongoDB successfully")

    def get_conversation(self, conversation_id: str, student_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Retrieve conversation history or create a new conversation
        Returns: (conversation_id, exchanges)
        """
        # If conversation_id is "new", create a new conversation
        if conversation_id == "new":
            new_id = str(uuid.uuid4())
            # Initialize new conversation with empty exchanges
            self.conversations.insert_one({
                "conversation_id": new_id,
                "student_id": student_id,
                "created_at": time.time(),
                "last_updated": time.time(),
                "exchanges": [],
                "current_scaffolding_level": settings.DEFAULT_SCAFFOLDING_LEVEL
            })
            logger.info(f"Created new conversation with ID: {new_id}")
            return new_id, []

        # Otherwise, retrieve existing conversation
        conversation = self.conversations.find_one({"conversation_id": conversation_id})

        if conversation and "exchanges" in conversation:
            logger.info(f"Retrieved conversation with ID: {conversation_id}")
            return conversation_id, conversation["exchanges"]

        # If conversation not found but ID was provided, create a new one with that ID
        self.conversations.insert_one({
            "conversation_id": conversation_id,
            "student_id": student_id,
            "created_at": time.time(),
            "last_updated": time.time(),
            "exchanges": [],
            "current_scaffolding_level": settings.DEFAULT_SCAFFOLDING_LEVEL
        })
        logger.info(f"Created new conversation with provided ID: {conversation_id}")
        return conversation_id, []

    def add_exchange(self,
                     conversation_id: str,
                     student_message: str,
                     assistant_message: str,
                     topic: str,
                     scaffolding_level: int,
                     student_id: Optional[str] = None,
                     metadata: Optional[Dict] = None):
        """Add a new exchange to the conversation"""
        exchange = {
            "student_message": student_message,
            "assistant_message": assistant_message,
            "timestamp": time.time(),
            "topic": topic,
            "scaffolding_level": scaffolding_level
        }

        if metadata:
            exchange["metadata"] = metadata

        # Update the conversation with the new exchange
        self.conversations.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"exchanges": exchange},
                "$set": {
                    "last_updated": time.time(),
                    "student_id": student_id,
                    "current_scaffolding_level": scaffolding_level
                }
            }
        )
        logger.info(f"Added new exchange to conversation: {conversation_id}")

    def get_current_scaffolding_level(self, conversation_id: str) -> int:
        """Get the current scaffolding level for a conversation"""
        conversation = self.conversations.find_one(
            {"conversation_id": conversation_id},
            {"current_scaffolding_level": 1}
        )

        if conversation and "current_scaffolding_level" in conversation:
            return conversation["current_scaffolding_level"]

        return settings.DEFAULT_SCAFFOLDING_LEVEL

    def update_scaffolding_level(self, conversation_id: str, level: int):
        """Update the scaffolding level for a conversation"""
        self.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"current_scaffolding_level": level}}
        )
        logger.info(f"Updated scaffolding level for conversation {conversation_id} to {level}")

    def get_student_conversations(self, student_id: str) -> List[Dict]:
        """Get all conversations for a student"""
        conversations = self.conversations.find(
            {"student_id": student_id},
            {"conversation_id": 1, "created_at": 1, "last_updated": 1, "current_scaffolding_level": 1}
        )

        return list(conversations)