FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/chroma_db

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""
ENV MONGODB_CONNECTION_STRING=""
ENV CHROMA_PERSIST_DIRECTORY="/app/data/chroma_db"
ENV LLM_MODEL="gpt-3.5-turbo"
ENV LLM_TEMPERATURE=0.2
ENV DEFAULT_SCAFFOLDING_LEVEL=2
ENV RETRIEVER_K=5
ENV API_HOST="0.0.0.0"
ENV API_PORT=8000
ENV LOG_LEVEL="INFO"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]