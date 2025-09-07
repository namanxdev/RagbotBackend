from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
import shutil
from typing import List
import uvicorn
import logging
from datetime import datetime
from Rag import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG PDF API", 
    description="API for uploading PDFs and asking questions",
    version="1.0.0"
)

# Get environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# CORS configuration
allowed_origins = [
    "http://localhost:3000",  # Local development
]

# Add production origins
if IS_PRODUCTION:
    allowed_origins.extend([
        "https://*.vercel.app",
        # Add your specific Vercel domain here once deployed
        # "https://your-app-name.vercel.app",
    ])
else:
    # Development - allow all origins
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

# Global RAG system instance
rag_system = None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    source_documents: List[str] = []

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    file_size_mb: float

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"Method: {request.method} | "
        f"URL: {request.url} | "
        f"Status: {response.status_code} | "
        f"Duration: {process_time:.2f}s"
    )
    
    return response

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it for RAG
    Maximum file size: 5MB
    """
    global rag_system
    
    logger.info(f"Received file upload request: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read file content to check size
    file_content = await file.read()
    file_size = len(file_content)
    file_size_mb = file_size / (1024 * 1024)
    
    # Validate file size (5MB limit)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size (5MB)"
        )
    
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write file content to temporary file
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Initialize RAG system with the uploaded PDF
        rag_system = RAGSystem(tmp_file_path)
        chunks_created = rag_system.get_chunks_count()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        logger.info(f"PDF processed successfully. Chunks created: {chunks_created}")
        
        return UploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created,
            file_size_mb=round(file_size_mb, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF {file.filename}: {str(e)}")
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded PDF
    """
    global rag_system
    
    logger.info(f"Received question: {request.question}")
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet. Please upload a PDF first.")
    
    try:
        answer, source_docs = rag_system.ask_question_with_sources(request.question)
        
        logger.info(f"Question answered successfully")
        
        return QuestionResponse(
            answer=answer,
            source_documents=source_docs
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint
    """
    try:
        # Basic checks
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "rag_system_loaded": rag_system is not None,
            "environment": ENVIRONMENT,
            "version": "1.0.0"
        }
        
        # Check environment variables
        required_env_vars = ["GEMINI_API_KEY", "DATABASE_URL"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            health_data["status"] = "unhealthy"
            health_data["missing_env_vars"] = missing_vars
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.delete("/reset")
async def reset_system():
    """
    Reset the RAG system (clear uploaded documents)
    """
    global rag_system
    logger.info("Resetting RAG system")
    rag_system = None
    return {"message": "RAG system reset successfully"}

@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "RAG PDF API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "endpoints": {
            "upload_pdf": "POST /upload-pdf",
            "ask_question": "POST /ask-question", 
            "health": "GET /health",
            "reset": "DELETE /reset",
            "docs": "GET /docs"
        },
        "max_file_size": "5MB",
        "status": "running"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
