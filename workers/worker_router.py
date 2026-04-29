import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workers.worker_service import load_model, generate_response

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "tinyllama-1.1b-chat"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str
    max_new_tokens: int = 256


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.post("/generate", response_model=AnswerResponse)
def generate(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = generate_response(request.question, request.max_new_tokens)
    return AnswerResponse(question=request.question, answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
