from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import filter_predictions_by_role, get_current_user
from database import get_db
from models import PredictionResult, User
from schemas import ReportSummary, Role, ScoreTrend, SegmentDistribution

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/summary", response_model=ReportSummary)
def summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.admin:
        total_users = db.query(func.count(User.id)).scalar()
    else:
        total_users = 1

    preds_q = db.query(PredictionResult)
    preds_q = filter_predictions_by_role(preds_q, current_user)

    total_preds = preds_q.count()
    avg_score = preds_q.with_entities(func.avg(PredictionResult.score)).scalar() or 0.0
    high = preds_q.filter(PredictionResult.segment == "high_intent").count()
    medium = preds_q.filter(PredictionResult.segment == "medium_intent").count()
    low = preds_q.filter(PredictionResult.segment == "low_intent").count()
    return ReportSummary(
        total_users=total_users,
        total_predictions=total_preds,
        avg_score=round(float(avg_score), 4),
        high_intent_count=high,
        medium_intent_count=medium,
        low_intent_count=low,
    )

@router.get("/segments", response_model=List[SegmentDistribution])
def segments(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = db.query(PredictionResult)
    q = filter_predictions_by_role(q, current_user)
    rows = (
        q.with_entities(
            PredictionResult.segment,
            func.count(PredictionResult.id).label("count"),
            func.avg(PredictionResult.score).label("avg_score"),
        )
        .group_by(PredictionResult.segment)
        .all()
    )
    return [SegmentDistribution(segment=r.segment, count=r.count, avg_score=round(float(r.avg_score), 4)) for r in rows]

@router.get("/trend", response_model=List[ScoreTrend])
def trend(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = db.query(PredictionResult)
    q = filter_predictions_by_role(q, current_user)
    rows = (
        q.with_entities(
            func.date(PredictionResult.predicted_at).label("date"),
            func.avg(PredictionResult.score).label("avg_score"),
            func.count(PredictionResult.id).label("count"),
        )
        .group_by(func.date(PredictionResult.predicted_at))
        .order_by(func.date(PredictionResult.predicted_at))
        .all()
    )
    return [ScoreTrend(date=str(r.date), avg_score=round(float(r.avg_score), 4), count=r.count) for r in rows]
