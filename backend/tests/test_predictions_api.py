import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import models
from main import app


class TestScoreAPI:

    def test_score_endpoint_returns_feature_details(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b = create_behavior(user_id=user.id)
        response = client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        assert response.status_code == 201
        data = response.json()
        assert "feature_details" in data
        assert isinstance(data["feature_details"], dict)
        assert len(data["feature_details"]) == 8
        for _, detail in data["feature_details"].items():
            assert "raw_value" in detail
            assert "normalized_value" in detail
            assert "weight" in detail

    def test_score_endpoint_historical_record_feature_details_empty(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b = create_behavior(user_id=user.id)
        from ml.scorer import classify_segment, score_behavior
        score, weights, _ = score_behavior(b.page_views, b.session_duration, b.clicks, b.email_opens, b.purchases, b.cart_adds, b.search_queries, b.days_since_last_visit)
        segment = classify_segment(score)
        old_pred = models.PredictionResult(
            user_id=user.id,
            behavior_id=b.id,
            score=score,
            segment=segment,
            feature_weights=weights,
            feature_details=None,
            score_bucket=int(time.time()),
        )
        db_session.add(old_pred)
        db_session.commit()
        db_session.refresh(old_pred)
        response = client.get(f"/predictions/{old_pred.id}", headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["feature_details"] == {}

    def test_duplicate_score_within_window_returns_409(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b = create_behavior(user_id=user.id)
        r1 = client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        assert r1.status_code == 201
        r2 = client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        assert r2.status_code == 409
        assert "Duplicate" in r2.json()["detail"]
        count = db_session.query(models.PredictionResult).filter_by(behavior_id=b.id).count()
        assert count == 1

    def test_different_behavior_scores_not_blocked(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b1 = create_behavior(user_id=user.id, page_views=10)
        b2 = create_behavior(user_id=user.id, page_views=20)
        r1 = client.post(f"/predictions/score/{b1.id}", headers=test_user_headers)
        r2 = client.post(f"/predictions/score/{b2.id}", headers=test_user_headers)
        assert r1.status_code == 201
        assert r2.status_code == 201
        count = db_session.query(models.PredictionResult).filter_by(user_id=user.id).count()
        assert count == 2

    def test_score_endpoint_returns_behavior_brief(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b = create_behavior(user_id=user.id, page_views=42, session_duration=999.0)
        response = client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        assert response.status_code == 201
        data = response.json()
        assert "behavior" in data
        assert data["behavior"] is not None
        assert data["behavior"]["page_views"] == 42
        assert data["behavior"]["session_duration"] == 999.0

    def test_list_predictions_batch_preload(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        for i in range(5):
            b = create_behavior(user_id=user.id, page_views=10 * (i + 1))
            client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        response = client.get("/predictions/", headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
        for pred in data:
            assert pred["behavior"] is not None
            assert "page_views" in pred["behavior"]
            assert pred["behavior"]["page_views"] > 0

    def test_viewer_role_cannot_score(self, client: TestClient, db_session: Session, create_behavior: callable):
        FAKE_BCRYPT_HASH = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
        viewer_password = "viewpass"
        viewer = models.User(
            email="viewer@example.com",
            username="viewer",
            hashed_password=FAKE_BCRYPT_HASH,
            role="viewer",
        )
        db_session.add(viewer)
        db_session.commit()
        db_session.refresh(viewer)

        import auth
        original_verify = auth.verify_password
        original_create = auth.create_access_token

        def _mock_verify(plain: str, hashed: str) -> bool:
            return plain == viewer_password

        def _mock_create_token(data: dict, role: str, expires_delta=None):
            return original_create(data, role, expires_delta)

        auth.verify_password = _mock_verify
        auth.create_access_token = _mock_create_token
        try:
            b = create_behavior(user_id=viewer.id)
            with TestClient(app) as c:
                r = c.post("/auth/login", data={"username": "viewer", "password": viewer_password})
                token = r.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                response = c.post(f"/predictions/score/{b.id}", headers=headers)
                assert response.status_code == 403
        finally:
            auth.verify_password = original_verify
            auth.create_access_token = original_create

    def test_batch_score_returns_behavior_data(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b1 = create_behavior(user_id=user.id, page_views=11)
        b2 = create_behavior(user_id=user.id, page_views=22)
        response = client.post(
            "/predictions/batch-score",
            json={"behavior_ids": [b1.id, b2.id]},
            headers=test_user_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["scored"] == 2
        for item in data["results"]:
            if item["status"] == "scored":
                pred = item["prediction"]
                assert "behavior" in pred
                assert pred["feature_details"] is not None

    def test_score_bucket_field_written(self, client: TestClient, db_session: Session, test_user_headers: dict, create_behavior: callable):
        user = db_session.query(models.User).filter_by(email="test@example.com").first()
        b = create_behavior(user_id=user.id)
        before = int(time.time())
        client.post(f"/predictions/score/{b.id}", headers=test_user_headers)
        after = int(time.time()) + 1
        pred = db_session.query(models.PredictionResult).filter_by(behavior_id=b.id).first()
        assert pred.score_bucket is not None
        assert before <= pred.score_bucket <= after
