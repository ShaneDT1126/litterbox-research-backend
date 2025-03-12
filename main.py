from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.core.logging import get_logger
from app.services.rag_engine import RAGEngine
from app.core.config import settings
import time
import traceback
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="LitterBox API",
    description="API for LitterBox, a scaffolding-based learning assistant for Computer Architecture",
    version="1.0.1",

)

# Add CORS middleware to allow Copilot Studio to access our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To be adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)

# Initialize RAG engine
rag_engine = RAGEngine()


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a student query and return a scaffolded response"""
    start_time = time.time()

    try:
        # Log incoming request
        logger.info(f"Processing query from student {request.student_id}: {request.query[:50]}...")

        # Process the query using RAG engine
        result = rag_engine.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            student_id=request.student_id
        )

        # Log processing time
        processing_time = time.time() - start_time
        logger.info(
            f"Query processed in {processing_time:.2f} seconds. Topic: {result['topic']}, Level: {result['scaffolding_level']}")

        return QueryResponse(
            response=result["response"],
            sources=[Source(**source) for source in result["sources"]],
            scaffolding_level=result["scaffolding_level"],
            topic=result["topic"],
            conversation_id=result["conversation_id"]
        )

    except Exception as e:
        # Log the error with traceback
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())

        # Return a user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again later."
        )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/v1/topics")
async def get_topics():
    """Get list of available topics"""
    try:
        # Return the list of topics from the RAG engine
        topics = list(rag_engine.topic_keywords.keys())
        return {"topics": topics}

    except Exception as e:
        logger.error(f"Error retrieving topics: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )