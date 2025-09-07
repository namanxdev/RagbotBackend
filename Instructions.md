# FastAPI RAG Backend Setup Instructions

This guide provides comprehensive instructions for setting up and running a FastAPI backend that allows PDF upload and question-answering using RAG (Retrieval-Augmented Generation).

## Project Overview

The backend consists of:
- `main.py`: FastAPI application with endpoints for PDF upload and Q&A
- `Rag.py`: RAGSystem class that handles PDF processing and question answering
- PostgreSQL database with pgvector extension for storing embeddings
- Google Gemini API for text generation and embeddings

## Prerequisites Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
IF NOT EXISTS Create a `.env` file in your project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/rag_database
```

### 3. Database Setup (Choose One Option)

#### Option A: Local PostgreSQL with Docker (Recommended)
```bash
# Run PostgreSQL with pgvector
docker run --name postgres-vector -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -d pgvector/pgvector:pg16

# Connect and create database
docker exec -it postgres-vector psql -U postgres
CREATE DATABASE rag_database;
\c rag_database;
CREATE EXTENSION vector;
\q
```

#### Option B: Cloud Database (Supabase/Neon)
1. Create account on Supabase (https://supabase.com/) or Neon (https://neon.tech/)
2. Create new project/database
3. Enable pgvector extension
4. Copy connection string to `.env` file

## API Endpoints

### 1. Upload PDF
- **Endpoint**: `POST /upload-pdf`
- **Purpose**: Upload and process a PDF file for RAG
- **Request**: Multipart form data with PDF file
- **Response**: 
  ```json
  {
    "message": "PDF uploaded and processed successfully",
    "filename": "document.pdf",
    "chunks_created": 25
  }
  ```

### 2. Ask Question
- **Endpoint**: `POST /ask-question`
- **Purpose**: Ask questions about uploaded PDF
- **Request**: 
  ```json
  {
    "question": "What is this document about?"
  }
  ```
- **Response**: 
  ```json
  {
    "answer": "This document discusses...",
    "source_documents": ["Document chunk 1...", "Document chunk 2..."]
  }
  ```

### 3. Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check if the API is running and if a PDF is loaded
- **Response**: 
  ```json
  {
    "status": "healthy",
    "rag_system_loaded": true
  }
  ```

### 4. Reset System
- **Endpoint**: `DELETE /reset`
- **Purpose**: Clear the current PDF and reset the system
- **Response**: 
  ```json
  {
    "message": "RAG system reset successfully"
  }
  ```

## Running the Backend

### 1. Start the Server
```bash
# Method 1: Direct execution
python main.py

# Method 2: Using uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the API
- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/redoc`

## Testing the API

### Using cURL

#### 1. Upload PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

#### 2. Ask Question
```bash
curl -X POST "http://localhost:8000/ask-question" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

### Using Python Requests
```python
import requests

# Upload PDF
with open("your_document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-pdf",
        files={"file": f}
    )
print(response.json())

# Ask question
response = requests.post(
    "http://localhost:8000/ask-question",
    json={"question": "What is this document about?"}
)
print(response.json())
```

## Frontend Integration

### HTML Form Example
```html
<!DOCTYPE html>
<html>
<head>
    <title>RAG PDF Q&A</title>
</head>
<body>
    <h1>PDF Question Answering System</h1>
    
    <!-- PDF Upload Form -->
    <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" id="pdfFile" accept=".pdf" required>
        <button type="submit">Upload PDF</button>
    </form>
    
    <!-- Question Form -->
    <div id="questionSection" style="display:none;">
        <h2>Ask Questions</h2>
        <input type="text" id="questionInput" placeholder="Enter your question">
        <button onclick="askQuestion()">Ask</button>
        <div id="answer"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('file', document.getElementById('pdfFile').files[0]);
            
            const response = await fetch('http://localhost:8000/upload-pdf', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                document.getElementById('questionSection').style.display = 'block';
                alert('PDF uploaded successfully!');
            }
        });
        
        async function askQuestion() {
            const question = document.getElementById('questionInput').value;
            const response = await fetch('http://localhost:8000/ask-question', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question})
            });
            
            const result = await response.json();
            document.getElementById('answer').innerHTML = `<p><strong>Answer:</strong> ${result.answer}</p>`;
        }
    </script>
</body>
</html>
```

## Architecture Details

### RAGSystem Class (`Rag.py`)
- **Initialization**: Loads PDF, creates embeddings, sets up vector store
- **PDF Processing**: Chunks documents using RecursiveCharacterTextSplitter
- **Vector Storage**: Uses PostgreSQL with pgvector for similarity search
- **Question Answering**: Implements RAG pipeline with LangGraph

### FastAPI Application (`main.py`)
- **File Upload**: Handles PDF uploads with validation
- **CORS Support**: Enables cross-origin requests for frontend integration
- **Error Handling**: Comprehensive error responses
- **State Management**: Maintains RAG system instance globally

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if PostgreSQL is running
   docker ps | grep postgres
   
   # Restart container if needed
   docker restart postgres-vector
   ```

2. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $GEMINI_API_KEY
   
   # Check .env file loading
   python -c "import dotenv; dotenv.load_dotenv(); import os; print(os.environ.get('GEMINI_API_KEY'))"
   ```

3. **Port Already in Use**
   ```bash
   # Find process using port 8000
   netstat -ano | findstr :8000
   
   # Kill process or use different port
   uvicorn main:app --port 8001
   ```

4. **PDF Processing Errors**
   - Ensure PDF is not corrupted
   - Check file size limits
   - Verify PDF is text-based (not image-only)

### Debugging Tips

1. **Enable Detailed Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Database Tables**
   ```sql
   \c rag_database;
   \dt  -- List tables
   SELECT COUNT(*) FROM document_embeddings_*;  -- Check document count
   ```

3. **Monitor API Requests**
   - Use FastAPI docs at `/docs` for interactive testing
   - Check server logs for detailed error messages

## Production Deployment

### 1. Environment Configuration
```env
# Production settings
ENVIRONMENT=production
DEBUG=False
DATABASE_URL=postgresql://prod_user:prod_pass@prod_host:5432/prod_db
CORS_ORIGINS=["https://yourdomain.com"]
```

### 2. Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Security Considerations
- Use environment variables for sensitive data
- Implement authentication/authorization
- Add rate limiting
- Validate file uploads thoroughly
- Use HTTPS in production

## Performance Optimization

1. **Chunk Size Tuning**: Adjust `chunk_size` and `chunk_overlap` in RAGSystem
2. **Vector Search**: Optimize similarity search parameters
3. **Caching**: Implement Redis for frequently asked questions
4. **Connection Pooling**: Configure PostgreSQL connection pools
5. **Async Processing**: Consider async document processing for large files

## Monitoring and Logging

```python
import logging
from fastapi import Request
import time

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"{request.method} {request.url} - {process_time:.2f}s")
    return response
```