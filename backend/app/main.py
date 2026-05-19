from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

from app.agent import run_agent

# ─── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Local Agents API",
    description="Backend for the local AI agent platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/response models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    model: str | None = None

class ChatResponse(BaseModel):
    answer: str

class ModelInfo(BaseModel):
    name: str
    size: float

# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Quick sanity check that the server is alive."""
    return {"status": "ok", "service": "local-agents-api"}

@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """List all Ollama models available on this machine."""
    try:
        result = ollama.list()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama. Is it running? Error: {e}"
        )

    try:
        models = []
        for m in result["models"]:
            model_name = m.get("model") or m.get("name") or "unknown"
            size = m.get("size", 0)
            models.append(ModelInfo(
                name=model_name,
                size=round(size / 1_000_000_000, 2),
            ))
        return models
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Got response from Ollama but could not parse it: {e}"
        )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a question to the research agent and get an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = run_agent(request.question)
        if answer is None:
            raise HTTPException(status_code=500, detail="Agent hit max steps without finishing")
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")    