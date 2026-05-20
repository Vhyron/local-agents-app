from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select
import ollama

from app.agent import run_agent, AgentConfig, available_tool_names
from app.db import init_db, get_session
from app.models import Agent, AgentCreate, AgentRead

# ─── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    init_db()
    yield
    # Nothing to clean up on shutdown for now

# ─── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Local Agents API",
    description="Backend for the local AI agent platform",
    version="0.1.0",
    lifespan=lifespan,
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
    agent_id: int | None = None

class ChatResponse(BaseModel):
    answer: str

class ModelInfo(BaseModel):
    name: str
    size: float

# ─── Health & models ───────────────────────────────────────────────────

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

# ─── Agents ────────────────────────────────────────────────────────────

@app.get("/agents", response_model=list[AgentRead])
def list_agents(session: Session = Depends(get_session)):
    """List all saved agents."""
    agents = session.exec(select(Agent)).all()
    return agents

@app.get("/agents/{agent_id}", response_model=AgentRead)
def get_agent(agent_id: int, session: Session = Depends(get_session)):
    """Get one agent by ID."""
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent

@app.post("/agents", response_model=AgentRead)
def create_agent(data: AgentCreate, session: Session = Depends(get_session)):
    """Create a new agent."""
    agent = Agent.model_validate(data)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent

@app.put("/agents/{agent_id}", response_model=AgentRead)
def update_agent(
    agent_id: int,
    data: AgentCreate,
    session: Session = Depends(get_session),
):
    """Update an existing agent."""
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    for key, value in data.model_dump().items():
        setattr(agent, key, value)

    from datetime import datetime
    agent.updated_at = datetime.utcnow()

    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: int, session: Session = Depends(get_session)):
    """Delete an agent."""
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    session.delete(agent)
    session.commit()
    return {"deleted": agent_id}

# ─── Tools ─────────────────────────────────────────────────────────────

class ToolInfo(BaseModel):
    name: str
    description: str

@app.get("/tools", response_model=list[ToolInfo])
def list_tools():
    """List all registered tools that agents can be configured to use."""
    tool_descriptions = {
        "web_search": "Search the web for current information",
        "read_url": "Fetch and extract the main text from a URL",
    }
    return [
        ToolInfo(name=name, description=tool_descriptions.get(name, ""))
        for name in available_tool_names()
    ]

# ─── Chat ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, session: Session = Depends(get_session)):
    """Send a question to an agent and get an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # If an agent_id was given, load its config; otherwise use defaults.
    if request.agent_id is not None:
        agent = session.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")

        config = AgentConfig(
            model=agent.model,
            system_prompt=agent.system_prompt,
            temperature=agent.temperature,
            enabled_tools=agent.enabled_tools,
        )
    else:
        config = AgentConfig()

    try:
        answer = run_agent(request.question, config=config)
        if answer is None:
            raise HTTPException(status_code=500, detail="Agent hit max steps without finishing")
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")