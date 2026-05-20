from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session

DB_PATH = Path(__file__).parent.parent / "local_agents.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    """Create all tables. Safe to call multiple times — only creates missing tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    Yield a database session for one request.
    FastAPI calls this via Depends() for each endpoint that needs DB access.
    """
    with Session(engine) as session:
        yield session