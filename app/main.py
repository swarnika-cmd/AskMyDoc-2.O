import os
import time
import shutil
import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.ingest import ingest_document
from app.retrieve import ask_question
from app.evaluate import evaluate_faithfulness
from app.database import delete_namespace
from app.metrics import metrics

app = FastAPI(title="AskMyDoc Backend (Self-Hosted)", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next):
    metrics.start_request()
    try:
        response = await call_next(request)
        return response
    finally:
        metrics.end_request()

# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

class ChatRequest(BaseModel):
    query: str
    collectionName: Optional[str] = None
    namespaces: Optional[List[str]] = None
    history: Optional[List[Dict[str, Any]]] = None

class DeleteRequest(BaseModel):
    namespace: str

class EvaluateRequest(BaseModel):
    query: str
    context: str
    answer: str

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "vector_store": os.getenv("VECTOR_STORE_TYPE", "chromadb")
    }

@app.get("/api/metrics")
async def get_metrics_endpoint():
    """Exposes real-time token throughput, memory utilization, latency, and queue depth metrics."""
    return metrics.get_metrics()

@app.post("/api/upload")
async def upload_document(
    document: UploadFile = File(...),
    collectionName: str = Form("DefaultCollection")
):
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()

    if not document.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    uploads_dir = "/tmp" if os.getenv("VERCEL") else "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    
    file_path = os.path.join(uploads_dir, f"{int(time.time())}_{document.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
            
        if os.path.getsize(file_path) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Uploaded file is empty (0 bytes). If you dragged it from a browser, try saving it to your computer first."
            )
            
        print(f"\n=== Starting Upload ===")
        print(f"File: {document.filename} ({os.path.getsize(file_path)} bytes)")
        print(f"Collection/Namespace: {collectionName}")
        print(f"Vector Store: {vector_store_type}")
        
        result = ingest_document(file_path, collectionName, document.filename)
        
        if result.get("chunksCount", 0) == 0:
            print("Upload completed but 0 chunks extracted (scanned/image PDF?)\n")
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this PDF. It may be a scanned/image-based document."
            )
            
        print("Upload completed successfully\n")
        return {"message": "File ingested successfully", "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nUpload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")
        
    try:
        target_namespaces = req.namespaces or ([req.collectionName] if req.collectionName else ["DefaultCollection"])
        history = req.history or []
        
        result = ask_question(req.query, target_namespaces, history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/delete")
async def delete_endpoint(req: DeleteRequest):
    namespace = req.namespace
    if not namespace:
        raise HTTPException(status_code=400, detail="Namespace is required")
        
    if namespace == 'demo-compliance':
        print("Prevented deletion of preloaded demo-compliance namespace.")
        return {"message": "Demo compliance namespace cannot be deleted."}
        
    try:
        success = delete_namespace(namespace)
        if success:
            print(f"Deleted all vectors from namespace: {namespace}")
            return {"message": f"Deleted namespace: {namespace}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete namespace: {namespace}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    try:
        result = evaluate_faithfulness(req.query, req.context, req.answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (served at root '/')
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
else:
    print("WARNING: 'public' directory not found. Static files will not be served.")
