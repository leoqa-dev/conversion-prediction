from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from database import get_db
from models import User, UserBehavior, PredictionResult
from schemas import PredictionOut, BatchScoreRequest, BatchScoreResponse, BatchScoreItem, Role
from auth import get_current_user, filter_predictions_by_role
from ml.scorer import score_behavior, classify_segment, WEIGHTS

router = APIRouter(prefix="/predictions", tags=["predictions"])

REQUIRED_DETAIL_FIELDS = {"raw_value", "normalized_value", "weight"}

def validate_feature_details(feature_details: Dict[str, Any]) -> None:
    if not isinstance(feature_details, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="feature_details must be a dictionary"
        )
    expected_features = set(WEIGHTS.keys())
    actual_features = set(feature_details.keys())
    if actual_features != expected_features:
        missing = expected_features - actual_features
        extra = actual_features - expected_features
        msg_parts = []
        if missing:
            msg_parts.append(f"missing features: {sorted(missing)}")
        if extra:
            msg_parts.append(f"unexpected features: {sorted(extra)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"feature_details structure incomplete: {', '.join(msg_parts)}"
        )
    for feat_name, detail in feature_details.items():
        if not isinstance(detail, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"feature '{feat_name}' detail must be a dictionary"
            )
        detail_fields = set(detail.keys())
        if detail_fields != REQUIRED_DETAIL_FIELDS:
            missing = REQUIRED_DETAIL_FIELDS - detail_fields
            extra = detail_fields - REQUIRED_DETAIL_FIELDS
            msg_parts = [f"feature '{feat_name}'"]
            if missing:
                msg_parts.append(f"missing fields: {sorted(missing)}")
            if extra:
                msg_parts.append(f"unexpected fields: {sorted(extra)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"feature_details structure incomplete: {', '.join(msg_parts)}"
            )

@router.post("/score/{behavior_id}", response_model=PredictionOut, status_code=201)
def score(behavior_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role cannot perform scoring operations"
        )
    b = db.query(UserBehavior).filter(UserBehavior.id == behavior_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Behavior record not found")
    probability, weights, feature_details = score_behavior(
        page_views=b.page_views,
        session_duration=b.session_duration,
        clicks=b.clicks,
        email_opens=b.email_opens,
        purchases=b.purchases,
        cart_adds=b.cart_adds,
        search_queries=b.search_queries,
        days_since_last_visit=b.days_since_last_visit,
    )
    validate_feature_details(feature_details)
    segment = classify_segment(probability)
    result = PredictionResult(
        user_id=b.user_id,
        behavior_id=b.id,
        score=probability,
        segment=segment,
        feature_weights=weights,
        feature_details=feature_details,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result

@router.get("/summary")
def predictions_summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = db.query(PredictionResult)
    q = filter_predictions_by_role(q, current_user)

    high = q.filter(PredictionResult.segment == "high_intent").count()
    medium = q.filter(PredictionResult.segment == "medium_intent").count()
    low = q.filter(PredictionResult.segment == "low_intent").count()
    total = high + medium + low

    return {
        "total": total,
        "segments": {
            "high_intent": high,
            "medium_intent": medium,
            "low_intent": low,
        }
    }

@router.get("/", response_model=List[PredictionOut])
def list_predictions(user_id: Optional[int] = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role can only access summary statistics, not individual records"
        )

    q = db.query(PredictionResult)
    q = filter_predictions_by_role(q, current_user)
    if user_id and current_user.role == Role.admin:
        q = q.filter(PredictionResult.user_id == user_id)
    return q.order_by(PredictionResult.predicted_at.desc()).all()

@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role can only access summary statistics, not individual records"
        )
    p = db.query(PredictionResult).filter(PredictionResult.id == prediction_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if current_user.role == Role.analyst and p.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return p

@router.post("/batch-score", response_model=BatchScoreResponse, status_code=201)
def batch_score(payload: BatchScoreRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role cannot perform scoring operations"
        )
    behavior_ids = list(dict.fromkeys(payload.behavior_ids))
    results: List[BatchScoreItem] = []
    scored_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        behaviors = db.query(UserBehavior).filter(UserBehavior.id.in_(behavior_ids)).all()
        behavior_map = {b.id: b for b in behaviors}

        existing = db.query(PredictionResult).filter(
            PredictionResult.behavior_id.in_(behavior_ids)
        ).all()
        existing_behavior_ids = {p.behavior_id for p in existing}

        created_predictions = []
        for bid in behavior_ids:
            if bid not in behavior_map:
                failed_count += 1
                results.append(BatchScoreItem(
                    behavior_id=bid,
                    status="failed",
                    message="Behavior record not found"
                ))
                continue

            if bid in existing_behavior_ids:
                skipped_count += 1
                existing_pred = next((p for p in existing if p.behavior_id == bid), None)
                results.append(BatchScoreItem(
                    behavior_id=bid,
                    status="skipped",
                    prediction=existing_pred,
                    message="Already scored, skipped"
                ))
                continue

            b = behavior_map[bid]
            try:
                probability, weights, feature_details = score_behavior(
                    page_views=b.page_views,
                    session_duration=b.session_duration,
                    clicks=b.clicks,
                    email_opens=b.email_opens,
                    purchases=b.purchases,
                    cart_adds=b.cart_adds,
                    search_queries=b.search_queries,
                    days_since_last_visit=b.days_since_last_visit,
                )
                validate_feature_details(feature_details)
                segment = classify_segment(probability)
                result = PredictionResult(
                    user_id=b.user_id,
                    behavior_id=b.id,
                    score=probability,
                    segment=segment,
                    feature_weights=weights,
                    feature_details=feature_details,
                )
                created_predictions.append((bid, result))
                scored_count += 1
            except HTTPException:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to score behavior {bid}: {str(e)}")

        for bid, pred in created_predictions:
            db.add(pred)
            results.append(BatchScoreItem(
                behavior_id=bid,
                status="scored",
                prediction=pred
            ))

        db.commit()

        for item in results:
            if item.status == "scored" and item.prediction:
                db.refresh(item.prediction)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch scoring failed, transaction rolled back: {str(e)}")

    return BatchScoreResponse(
        total=len(behavior_ids),
        scored=scored_count,
        skipped=skipped_count,
        failed=failed_count,
        results=results
    )
