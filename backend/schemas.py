from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, EmailStr, field_serializer

class Role(str, Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: Optional[Role] = Role.analyst

class UserOut(BaseModel):
    id: int
    email: str
    username: str
    role: Role
    created_at: datetime
    model_config = {"from_attributes": True}

class Token(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class BehaviorCreate(BaseModel):
    user_id: int
    page_views: int = 0
    session_duration: float = 0.0
    clicks: int = 0
    email_opens: int = 0
    purchases: int = 0
    cart_adds: int = 0
    search_queries: int = 0
    days_since_last_visit: float = 0.0

class BehaviorUpdate(BaseModel):
    page_views: Optional[int] = None
    session_duration: Optional[float] = None
    clicks: Optional[int] = None
    email_opens: Optional[int] = None
    purchases: Optional[int] = None
    cart_adds: Optional[int] = None
    search_queries: Optional[int] = None
    days_since_last_visit: Optional[float] = None

class BehaviorOut(BaseModel):
    id: int
    user_id: int
    page_views: int
    session_duration: float
    clicks: int
    email_opens: int
    purchases: int
    cart_adds: int
    search_queries: int
    days_since_last_visit: float
    recorded_at: datetime
    model_config = {"from_attributes": True}

class FeatureDetail(BaseModel):
    raw_value: Any
    normalized_value: float
    weight: float

class PredictionOut(BaseModel):
    id: int
    user_id: int
    behavior_id: Optional[int]
    score: float
    segment: str
    feature_weights: Optional[dict]
    feature_details: Optional[Dict[str, FeatureDetail]] = None
    predicted_at: datetime
    model_config = {"from_attributes": True}

    @field_serializer("feature_details")
    def serialize_feature_details(self, v: Optional[Dict[str, FeatureDetail]]) -> Dict[str, Any]:
        if v is None:
            return {}
        return {k: fd.model_dump() for k, fd in v.items()}

class ReportSummary(BaseModel):
    total_users: int
    total_predictions: int
    avg_score: float
    high_intent_count: int
    medium_intent_count: int
    low_intent_count: int

class SegmentDistribution(BaseModel):
    segment: str
    count: int
    avg_score: float

class ScoreTrend(BaseModel):
    date: str
    avg_score: float
    count: int

class BatchScoreRequest(BaseModel):
    behavior_ids: List[int]

class BatchScoreItem(BaseModel):
    behavior_id: int
    status: str
    prediction: Optional[PredictionOut] = None
    message: Optional[str] = None

class BatchScoreResponse(BaseModel):
    total: int
    scored: int
    skipped: int
    failed: int
    results: List[BatchScoreItem]
