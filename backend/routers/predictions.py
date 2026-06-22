import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import filter_predictions_by_role, get_current_user
from database import get_db, get_query_count, reset_query_counter
from ml.scorer import WEIGHTS, classify_segment, score_behavior
from models import PredictionResult, User, UserBehavior
from schemas import BatchScoreItem, BatchScoreRequest, BatchScoreResponse, PredictionOut, Role

logger = logging.getLogger("predictions.router")

SCORE_WINDOW_SECONDS = 1


def _current_score_bucket() -> int:
    return int(time.time() // SCORE_WINDOW_SECONDS)

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
    api_start = time.perf_counter()
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
    score_bucket = _current_score_bucket()
    result = PredictionResult(
        user_id=b.user_id,
        behavior_id=b.id,
        score=probability,
        segment=segment,
        feature_weights=weights,
        feature_details=feature_details,
        score_bucket=score_bucket,
    )
    db.add(result)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        api_elapsed_ms = (time.perf_counter() - api_start) * 1000
        logger.warning(
            "Duplicate score blocked by unique constraint: user_id=%d, behavior_id=%d, score_bucket=%d (window=%ds) | total_api_time=%.3f ms",
            b.user_id, b.id, score_bucket, SCORE_WINDOW_SECONDS, api_elapsed_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Duplicate scoring request detected: the same behavior (id={b.id}) "
                f"has already been scored for user (id={b.user_id}) within the last "
                f"{SCORE_WINDOW_SECONDS} second(s). Please wait a moment before retrying."
            ),
        ) from None
    db.refresh(result)
    result.behavior = b
    api_elapsed_ms = (time.perf_counter() - api_start) * 1000
    logger.info(
        "POST /predictions/score/%d completed in %.3f ms | user_id=%d | score=%.4f | segment=%s",
        behavior_id, api_elapsed_ms, b.user_id, probability, segment,
    )
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

    reset_query_counter()

    q = db.query(PredictionResult)
    q = filter_predictions_by_role(q, current_user)
    if user_id and current_user.role == Role.admin:
        q = q.filter(PredictionResult.user_id == user_id)
    predictions = q.order_by(PredictionResult.predicted_at.desc()).all()

    behavior_ids = [p.behavior_id for p in predictions if p.behavior_id is not None]
    if behavior_ids:
        behaviors = db.query(UserBehavior).filter(UserBehavior.id.in_(behavior_ids)).all()
        behavior_map = {b.id: b for b in behaviors}
    else:
        behavior_map = {}

    for p in predictions:
        if p.behavior_id is not None and p.behavior_id in behavior_map:
            p.behavior = behavior_map[p.behavior_id]

    query_count = get_query_count()
    logger.info(
        "list_predictions: returned %d predictions, total SQL queries=%d (constant, not N+1)",
        len(predictions), query_count,
    )

    return predictions

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
    if p.behavior_id is not None:
        b = db.query(UserBehavior).filter(UserBehavior.id == p.behavior_id).first()
        if b:
            p.behavior = b
    return p

@router.post("/batch-score", response_model=BatchScoreResponse, status_code=201)
def batch_score(payload: BatchScoreRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == Role.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role cannot perform scoring operations"
        )
    api_start = time.perf_counter()
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

        failed_items = []
        skipped_items = []
        created_predictions = []
        for bid in behavior_ids:
            if bid not in behavior_map:
                failed_count += 1
                failed_items.append(BatchScoreItem(
                    behavior_id=bid,
                    status="failed",
                    message="Behavior record not found"
                ))
                continue

            if bid in existing_behavior_ids:
                skipped_count += 1
                existing_pred = next((p for p in existing if p.behavior_id == bid), None)
                if existing_pred and existing_pred.behavior_id is not None:
                    existing_pred.behavior = behavior_map.get(existing_pred.behavior_id)
                skipped_items.append(BatchScoreItem(
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
                score_bucket = _current_score_bucket()
                result = PredictionResult(
                    user_id=b.user_id,
                    behavior_id=b.id,
                    score=probability,
                    segment=segment,
                    feature_weights=weights,
                    feature_details=feature_details,
                    score_bucket=score_bucket,
                )
                created_predictions.append((bid, result, b))
                scored_count += 1
            except HTTPException:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to score behavior {bid}: {str(e)}")

        for _, pred, _ in created_predictions:
            db.add(pred)

        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            api_elapsed_ms = (time.perf_counter() - api_start) * 1000
            conflict_buckets = {(p.user_id, p.behavior_id, p.score_bucket) for _, p, _ in created_predictions}
            logger.warning(
                "Batch scoring duplicate blocked by unique constraint: %d candidate(s), "
                "user_id/behavior_id/bucket tuples=%s (window=%ds) | total_api_time=%.3f ms",
                len(conflict_buckets), sorted(conflict_buckets), SCORE_WINDOW_SECONDS, api_elapsed_ms,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Duplicate scoring detected during batch operation: one or more behavior records "
                    f"have already been scored within the last {SCORE_WINDOW_SECONDS} second(s). "
                    "Please wait a moment before retrying."
                ),
            ) from None

        scored_items = []
        for bid, pred, b in created_predictions:
            db.refresh(pred)
            pred.behavior = b
            scored_items.append(BatchScoreItem(
                behavior_id=bid,
                status="scored",
                prediction=pred
            ))

        results = failed_items + skipped_items + scored_items

    except Exception as e:
        db.rollback()
        api_elapsed_ms = (time.perf_counter() - api_start) * 1000
        logger.error(
            "POST /predictions/batch-score failed in %.3f ms | total=%d, error=%s",
            api_elapsed_ms, len(behavior_ids), str(e),
        )
        raise HTTPException(status_code=500, detail=f"Batch scoring failed, transaction rolled back: {str(e)}")

    api_elapsed_ms = (time.perf_counter() - api_start) * 1000
    logger.info(
        "POST /predictions/batch-score completed in %.3f ms | total=%d, scored=%d, skipped=%d, failed=%d",
        api_elapsed_ms, len(behavior_ids), scored_count, skipped_count, failed_count,
    )
    return BatchScoreResponse(
        total=len(behavior_ids),
        scored=scored_count,
        skipped=skipped_count,
        failed=failed_count,
        results=results
    )
