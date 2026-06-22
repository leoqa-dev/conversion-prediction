import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger("sql_query_counter")

DATABASE_URL = "sqlite:///./conversion.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

_query_counter = 0

@event.listens_for(engine, "before_cursor_execute")
def _on_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    global _query_counter
    _query_counter += 1
    logger.info("[SQL #%d] %s", _query_counter, statement.strip()[:200])

def reset_query_counter() -> None:
    global _query_counter
    _query_counter = 0

def get_query_count() -> int:
    return _query_counter

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
