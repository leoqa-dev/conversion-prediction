import sys
sys.path.insert(0, '.')
from ml.scorer import score_behavior, classify_segment

print("=== Test 1: 所有行为字段为零（新注册用户） ===")
score, weights = score_behavior(
    page_views=0, session_duration=0.0, clicks=0, email_opens=0,
    purchases=0, cart_adds=0, search_queries=0, days_since_last_visit=0.0
)
segment = classify_segment(score)
print(f"  score={score}, segment={segment}")
print(f"  all weights zero={all(v == 0.0 for v in weights.values())}")
assert score == 0.0, f"Expected score 0.0, got {score}"
assert segment == "low_intent", f"Expected low_intent, got {segment}"
print("  PASSED")

print("\n=== Test 2: 有行为数据的用户 - 验证不受影响 ===")
score2, weights2 = score_behavior(
    page_views=10, session_duration=300.0, clicks=5, email_opens=2,
    purchases=1, cart_adds=3, search_queries=4, days_since_last_visit=1.0
)
segment2 = classify_segment(score2)
print(f"  score={score2}, segment={segment2}")
assert score2 > 0.0, f"Expected score > 0, got {score2}"
print("  PASSED")

print("\n=== Test 3: 部分字段非零 - 也应该正常评分 ===")
score3, _ = score_behavior(
    page_views=1, session_duration=0.0, clicks=0, email_opens=0,
    purchases=0, cart_adds=0, search_queries=0, days_since_last_visit=0.0
)
print(f"  score={score3}, segment={classify_segment(score3)}")
assert score3 > 0.0
print("  PASSED")

print("\n=== Test 4: 仅 days_since_last_visit 非零 - 视为零行为 ===")
score4, _ = score_behavior(
    page_views=0, session_duration=0.0, clicks=0, email_opens=0,
    purchases=0, cart_adds=0, search_queries=0, days_since_last_visit=5.0
)
print(f"  score={score4}, segment={classify_segment(score4)}")
assert score4 == 0.0
print("  PASSED")

print("\n=== All tests passed! ===")
