from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="analyst", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    behaviors = relationship("UserBehavior", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("PredictionResult", back_populates="user", cascade="all, delete-orphan")

class UserBehavior(Base):
    __tablename__ = "user_behaviors"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    page_views = Column(Integer, default=0)
    session_duration = Column(Float, default=0.0)
    clicks = Column(Integer, default=0)
    email_opens = Column(Integer, default=0)
    purchases = Column(Integer, default=0)
    cart_adds = Column(Integer, default=0)
    search_queries = Column(Integer, default=0)
    days_since_last_visit = Column(Float, default=0.0)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="behaviors")

class PredictionResult(Base):
    __tablename__ = "prediction_results"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "behavior_id",
            "score_bucket",
            name="uq_prediction_user_behavior_bucket",
        ),
    )
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    behavior_id = Column(Integer, ForeignKey("user_behaviors.id"), nullable=True)
    score = Column(Float, nullable=False)
    segment = Column(String, nullable=False)
    feature_weights = Column(JSON, nullable=True)
    feature_details = Column(JSON, nullable=True)
    score_bucket = Column(Integer, nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")
    behavior = relationship("UserBehavior", lazy="noload")
