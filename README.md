# LitterBox: Computer Architecture Learning Companion

LitterBox is an AI-powered learning companion specialized in Computer Architecture that uses scaffolding techniques to guide students through complex topics. Rather than providing direct answers, LitterBox helps students discover knowledge through guided exploration.

## Features

- **Adaptive Scaffolding**: Provides the right level of support based on student understanding
- **Guided Discovery**: Helps students build knowledge through exploration rather than direct instruction
- **Topic-Specific Guidance**: Specialized in Computer Architecture topics
- **Conversation Memory**: Maintains context across conversation sessions
- **Natural Interaction**: Creates conversational, varied responses that avoid repetitive patterns

## Architecture

LitterBox uses a single template approach with contextual adaptation to generate appropriate responses based on:
- Scaffolding level (1-3)
- Topic detection
- Conversation history
- Special scenarios (confusion, acknowledgment, etc.)

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository: 
2. Create a virtual environment: python -m venv venv and then source venv/bin/activate (For Windows, do: venv\Scripts\activate)
3. Install dependencies: pip install -r requirements.txt
4. Setup environment variables
5. Run the application: python main.py


### Docker Setup

1. Build the Docker image: docker build -t litterbox:latest .
2. Run the container: docker run -p 8000:8000 --env-file .env litterbox:latest

## API Documentation

### Process Query
Processes a student query and returns a scaffolded response

**Endpoint**: `POST /api/v1/query`

**Request Body**: 
```json
{
"query" : "str",
"conversation" : "str",
"student_id" : "str"
}
```

**Response (sample):**
```json
{
  "response": "Cache coherence addresses a challenge in multiprocessor systems...",
  "sources": [
    {
      "content": "Excerpt from source...",
      "topic": "Cache Coherence",
      "confidence": 0.85,
      "metadata": {
        "source": "Computer Architecture: A Quantitative Approach",
        "page": 123
      }
    }
  ],
  "scaffolding_level": 2,
  "topic": "Cache Coherence",
  "conversation_id": "conversation-identifier"
}
```


