import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from ml.scorer import NORMALIZERS, WEIGHTS, classify_segment, score_behavior


class TestScoreBehavior:

    def test_zero_behavior_returns_zero(self):
        score, weights, details = score_behavior(0, 0, 0, 0, 0, 0, 0, 0)
        assert score == 0.0
        assert all(v == 0.0 for v in weights.values())
        assert isinstance(details, dict)
        assert set(details.keys()) == set(WEIGHTS.keys())

    def test_score_range(self):
        for _ in range(100):
            import random
            score, _, _ = score_behavior(
                random.randint(0, 200),
                random.uniform(0, 7200),
                random.randint(0, 400),
                random.randint(0, 60),
                random.randint(0, 20),
                random.randint(0, 40),
                random.randint(0, 100),
                random.uniform(0, 60),
            )
            assert 0.0 <= score <= 1.0

    def test_feature_details_structure(self):
        _, _, details = score_behavior(50, 1800, 100, 15, 5, 10, 25, 3)
        required_fields = {"raw_value", "normalized_value", "weight"}
        for _, detail in details.items():
            assert isinstance(detail, dict)
            assert set(detail.keys()) == required_fields

    def test_feature_details_values_match_input(self):
        inputs = {
            "page_views": 50,
            "session_duration": 1800,
            "clicks": 100,
            "email_opens": 15,
            "purchases": 5,
            "cart_adds": 10,
            "search_queries": 25,
            "days_since_last_visit": 3,
        }
        _, _, details = score_behavior(**inputs)
        for feat, expected_raw in inputs.items():
            assert details[feat]["raw_value"] == expected_raw

    def test_feature_details_normalized_capped_at_1(self):
        _, _, details = score_behavior(9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999)
        for _, detail in details.items():
            assert 0.0 <= detail["normalized_value"] <= 1.0

    def test_feature_details_weights_match_config(self):
        _, _, details = score_behavior(10, 100, 5, 3, 1, 2, 4, 1)
        for feat, detail in details.items():
            assert detail["weight"] == WEIGHTS[feat]

    def test_higher_purchases_increases_score(self):
        score_low, _, _ = score_behavior(10, 100, 5, 3, 1, 2, 4, 1)
        score_high, _, _ = score_behavior(10, 100, 5, 3, 10, 2, 4, 1)
        assert score_high > score_low

    def test_days_since_last_visit_penalizes_score(self):
        score_fresh, _, _ = score_behavior(10, 100, 5, 3, 1, 2, 4, 1)
        score_stale, _, _ = score_behavior(10, 100, 5, 3, 1, 2, 4, 30)
        assert score_fresh > score_stale

    def test_consistent_input_produces_consistent_output(self):
        s1, w1, d1 = score_behavior(50, 1800, 100, 15, 5, 10, 25, 3)
        s2, w2, d2 = score_behavior(50, 1800, 100, 15, 5, 10, 25, 3)
        assert s1 == s2
        assert w1 == w2
        assert d1 == d2


class TestClassifySegment:

    def test_high_intent(self):
        assert classify_segment(0.80) == "high_intent"
        assert classify_segment(0.70) == "high_intent"

    def test_medium_intent(self):
        assert classify_segment(0.60) == "medium_intent"
        assert classify_segment(0.40) == "medium_intent"

    def test_low_intent(self):
        assert classify_segment(0.30) == "low_intent"
        assert classify_segment(0.0) == "low_intent"

    def test_boundary_values(self):
        assert classify_segment(0.6999) == "medium_intent"
        assert classify_segment(0.7000) == "high_intent"
        assert classify_segment(0.3999) == "low_intent"
        assert classify_segment(0.4000) == "medium_intent"


class TestWeightSanity:

    def test_purchases_has_highest_weight(self):
        assert WEIGHTS["purchases"] == max(WEIGHTS.values())

    def test_days_since_last_visit_is_negative(self):
        assert WEIGHTS["days_since_last_visit"] < 0

    def test_all_normalizers_are_positive(self):
        assert all(v > 0 for v in NORMALIZERS.values())
