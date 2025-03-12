from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Source(BaseModel):
    """Source document for a response"""
    content: str
    topic: str
    confidence: float = 0.8
    metadata: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    conversation_id: str = "new"
    student_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    sources: List[Source] = []
    scaffolding_level: int
    topic: str
    conversation_id: str