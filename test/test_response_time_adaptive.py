"""
Response-Time Adaptive Scheduling — Unit Test
==============================================
Tests the new _calculate_time_adjustment logic in BehaviorTracker
(no server needed — imports directly from the codebase).

Run: python test/test_response_time_adaptive.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from src.features.reminder_system.behavior_tracker import BehaviorTracker
from src.features.reminder_system.reminder_models import ReminderInteraction, InteractionType

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

total = 0
passed = 0

def check(label, condition, got=None, expected=None):
    global total, passed
    total += 1
    if condition:
        passed += 1
        print(f"  {PASS}  {label}")
    else:
        detail = f"  → got={got!r}, expected={expected!r}" if got is not None else ""
        print(f"  {FAIL}  {label}{detail}")

def make_interactions(count, response_delay_seconds, interaction_type=InteractionType.CONFIRMED):
    """Build a list of ReminderInteraction with given response_time_seconds."""
    interactions = []
    for i in range(count):
        t = datetime.now() - timedelta(days=count - i)
        interactions.append(ReminderInteraction(
            reminder_id=f"r_{i}",
            user_id="margaret_test",
            reminder_category="medication",
            interaction_type=interaction_type,
            interaction_time=t,
            response_time_seconds=response_delay_seconds,
            cognitive_risk_score=0.2
        ))
    return interactions

tracker = BehaviorTracker()

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 1: Margaret always responds 12 minutes late")
print("  Expected: system recommends shifting alarm -12 minutes (earlier)")
print("================================================================")

interactions_12min = make_interactions(count=10, response_delay_seconds=720)  # 720s = 12 min
adj = tracker._calculate_time_adjustment(interactions_12min)
print(f"  time_adjustment_minutes = {adj}")
check("Recommend shifting alarm EARLIER (negative value)", adj < 0, got=adj, expected="< 0")
check("Adjustment close to -12 minutes", -15 <= adj <= -10, got=adj, expected="-10 to -15")

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 2: Margaret responds in 3 minutes (good, no shift needed)")
print("  Expected: 0 (no adjustment)")
print("================================================================")

interactions_3min = make_interactions(count=10, response_delay_seconds=180)  # 180s = 3 min
adj2 = tracker._calculate_time_adjustment(interactions_3min)
print(f"  time_adjustment_minutes = {adj2}")
check("No adjustment for < 5 min avg delay", adj2 == 0, got=adj2, expected=0)

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 3: Only 5 interactions (cold start, need ≥7 for adjustment)")
print("  Expected: 0")
print("================================================================")

interactions_few = make_interactions(count=5, response_delay_seconds=720)
adj3 = tracker._calculate_time_adjustment(interactions_few)
print(f"  time_adjustment_minutes = {adj3}")
check("No adjustment with <7 interactions (cold start)", adj3 == 0, got=adj3, expected=0)

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 4: Margaret responds 30 minutes late (extreme case)")
print("  Expected: -30, capped at max -60")
print("================================================================")

interactions_30min = make_interactions(count=10, response_delay_seconds=1800)  # 1800s = 30 min
adj4 = tracker._calculate_time_adjustment(interactions_30min)
print(f"  time_adjustment_minutes = {adj4}")
check("Recommend -30 min shift for 30-min late responder", -35 <= adj4 <= -25, got=adj4, expected="-25 to -35")

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 5: Fallback — IGNORED interactions, no response_time_seconds")
print("  Expected: +30 (shift later — user isn't home at this hour)")
print("================================================================")

ignored_interactions = make_interactions(count=10, response_delay_seconds=None, 
                                          interaction_type=InteractionType.IGNORED)
for i in ignored_interactions:
    i.response_time_seconds = None   # simulate no response time data
    # All at same hour to trigger fallback
    i.interaction_time = i.interaction_time.replace(hour=8)

adj5 = tracker._calculate_time_adjustment(ignored_interactions)
print(f"  time_adjustment_minutes = {adj5}")
check("Fallback +30 when consistently IGNORED at same hour", adj5 == 30, got=adj5, expected=30)

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  TEST 6: Full BehaviorPattern — 1 week of 12-min-late responses")
print("  Expected: recommended_time_adjustment_minutes ≈ -12")
print("================================================================")

for interaction in make_interactions(count=10, response_delay_seconds=720):
    tracker.log_interaction(interaction)

pattern = tracker.get_user_behavior_pattern(user_id="margaret_test", category="medication", days=30)
print(f"  pattern.recommended_time_adjustment_minutes = {pattern.recommended_time_adjustment_minutes}")
print(f"  pattern.avg_response_time_seconds           = {pattern.avg_response_time_seconds}")
check("Full pattern correctly computes negative adjustment",
      pattern.recommended_time_adjustment_minutes < 0,
      got=pattern.recommended_time_adjustment_minutes, expected="< 0")
check("Pattern avg_response_time_seconds ≈ 720",
      pattern.avg_response_time_seconds is not None and abs(pattern.avg_response_time_seconds - 720) < 60,
      got=pattern.avg_response_time_seconds, expected="≈720")

# ─────────────────────────────────────────────────────────────────────────────
print("\n================================================================")
print("  FINAL RESULTS")
print("================================================================")
pct = passed / total * 100
grade = "EXCELLENT" if pct >= 90 else "GOOD" if pct >= 75 else "PARTIAL"
print(f"\n  Score: {passed}/{total}  ({pct:.0f}%)  — {grade}")
if pct < 100:
    print(f"\n  Note: FAIL items show what was got vs expected")
print()

sys.exit(0 if pct >= 75 else 1)
