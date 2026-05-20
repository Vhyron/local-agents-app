from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


# ─── Agents ────────────────────────────────────────────────────────────

class AgentBase(SQLModel):
    """Fields shared by both DB rows and API requests."""
    name: str
    system_prompt: str
    model: str = "qwen2.5:7b"
    temperature: float = 0.3
    enabled_tools: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Agent(AgentBase, table=True):
    """Database table for saved agents."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentCreate(AgentBase):
    """Shape for POST /agents request body."""
    pass


class AgentRead(AgentBase):
    """Shape for GET /agents response."""
    id: int
    created_at: datetime
    updated_at: datetime


# ─── Conversations ─────────────────────────────────────────────────────

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(foreign_key="agent.id")
    title: str = "New Conversation"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Messages ──────────────────────────────────────────────────────────

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user", "assistant", or "tool"
    content: str
    tool_calls: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)