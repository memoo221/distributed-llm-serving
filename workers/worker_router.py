import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workers.worker_service import WorkerNode
from workers.inference import generate_response

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "tinyllama-1.1b-chat"),
)

# Global worker instance
_worker: WorkerNode = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker
    _worker = WorkerNode(
        model_path=MODEL_PATH,
        device="cpu"  # CPU only for this container
    )
    master_url = os.getenv("MASTER_URL")
    if master_url:
        _worker.start_heartbeat(master_url, interval_sec=5.0)
    yield
    if master_url:
        _worker.stop_heartbeat()
    print(f"[worker_router] Worker initialized with model: {MODEL_PATH}")
    yield
    print("[worker_router] Worker shutdown")


app = FastAPI(lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str
    max_new_tokens: int = 256


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.post("/generate", response_model=AnswerResponse)
def generate(request: QuestionRequest):
    global _worker
    if _worker is None:
        raise HTTPException(status_code=500, detail="Worker not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer = _worker.generate(request.question, max_new_tokens=request.max_new_tokens)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/health")
def health():
    global _worker
    if _worker is None:
        return {"status": "initializing"}
    return {"status": "ok"}
