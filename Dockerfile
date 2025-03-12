FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser

# Create necessary directories
RUN mkdir -p data/chroma_db && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""
ENV MONGODB_CONNECTION_STRING=""
ENV CHROMA_PERSIST_DIRECTORY="/app/data/chroma_db"
ENV LLM_MODEL="gpt-4o-mini"
ENV LLM_TEMPERATURE=0.2
ENV DEFAULT_SCAFFOLDING_LEVEL=2
ENV RETRIEVER_K=5
ENV API_HOST="0.0.0.0"
ENV API_PORT=8000
ENV LOG_LEVEL="INFO"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]