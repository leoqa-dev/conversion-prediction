import numpy as np
from typing import Dict, Tuple

WEIGHTS = {
    "purchases": 0.35,
    "cart_adds": 0.20,
    "email_opens": 0.15,
    "session_duration": 0.12,
    "page_views": 0.08,
    "clicks": 0.06,
    "search_queries": 0.04,
    "days_since_last_visit": -0.10,
}

NORMALIZERS = {
    "purchases": 10.0,
    "cart_adds": 20.0,
    "email_opens": 30.0,
    "session_duration": 3600.0,
    "page_views": 100.0,
    "clicks": 200.0,
    "search_queries": 50.0,
    "days_since_last_visit": 30.0,
}

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def score_behavior(
    page_views: int,
    session_duration: float,
    clicks: int,
    email_opens: int,
    purchases: int,
    cart_adds: int,
    search_queries: int,
    days_since_last_visit: float,
) -> Tuple[float, Dict[str, float]]:
    raw = {
        "purchases": purchases,
        "cart_adds": cart_adds,
        "email_opens": email_opens,
        "session_duration": session_duration,
        "page_views": page_views,
        "clicks": clicks,
        "search_queries": search_queries,
        "days_since_last_visit": days_since_last_visit,
    }

    active_behavior_sum = (
        page_views + session_duration + clicks + email_opens
        + purchases + cart_adds + search_queries
    )
    if active_behavior_sum == 0:
        feature_weights = {k: 0.0 for k in WEIGHTS}
        return 0.0, feature_weights

    linear = sum(
        WEIGHTS[k] * min(raw[k] / NORMALIZERS[k], 1.0)
        for k in WEIGHTS
    )
    score = round(float(_sigmoid(linear * 6 - 2)), 4)
    feature_weights = {k: round(WEIGHTS[k] * min(raw[k] / NORMALIZERS[k], 1.0), 4) for k in WEIGHTS}
    return score, feature_weights

def classify_segment(score: float) -> str:
    if score >= 0.70:
        return "high_intent"
    elif score >= 0.40:
        return "medium_intent"
    return "low_intent"
