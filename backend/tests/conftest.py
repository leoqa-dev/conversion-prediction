import os
import sys
from typing import Any, Dict, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

import models
from auth import hash_password
from database import Base, get_db
from main import app

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function", autouse=True)
def setup_db() -> Generator[None, None, None]:
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


FAKE_BCRYPT_HASH = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"


@pytest.fixture
def test_user_headers(db_session: Session) -> Dict[str, str]:
    password = "testpass123"
    user = models.User(
        email="test@example.com",
        username="testuser",
        hashed_password=FAKE_BCRYPT_HASH,
        role="analyst",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    import auth
    from auth import create_access_token, verify_password
    original_verify = auth.verify_password

    def _mock_verify(plain: str, hashed: str) -> bool:
        return plain == password

    auth.verify_password = _mock_verify
    try:
        token = create_access_token(data={"sub": str(user.id)}, role=user.role)
    finally:
        auth.verify_password = original_verify

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(db_session: Session) -> Dict[str, str]:
    password = "adminpass123"
    user = models.User(
        email="admin@example.com",
        username="admin",
        hashed_password=FAKE_BCRYPT_HASH,
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    import auth
    from auth import create_access_token, verify_password
    original_verify = auth.verify_password

    def _mock_verify(plain: str, hashed: str) -> bool:
        return plain == password

    auth.verify_password = _mock_verify
    try:
        token = create_access_token(data={"sub": str(user.id)}, role=user.role)
    finally:
        auth.verify_password = original_verify

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def create_behavior(db_session: Session) -> callable:
    def _create_behavior(user_id: int, **kwargs) -> models.UserBehavior:
        defaults = {
            "page_views": 10,
            "session_duration": 100.0,
            "clicks": 5,
            "email_opens": 3,
            "purchases": 1,
            "cart_adds": 2,
            "search_queries": 4,
            "days_since_last_visit": 1.0,
        }
        defaults.update(kwargs)
        b = models.UserBehavior(user_id=user_id, **defaults)
        db_session.add(b)
        db_session.commit()
        db_session.refresh(b)
        return b
    return _create_behavior
