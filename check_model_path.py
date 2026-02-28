"""Quick check to confirm which model folder is being used"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.game.model_registry import RISK_CLASSIFIER_DIR

print("\n" + "=" * 70)
print("📁 MODEL PATH VERIFICATION")
print("=" * 70)
print(f"\n🔍 Backend is loading models from:")
print(f"   {RISK_CLASSIFIER_DIR}")
print(f"\n📄 Files in this directory:")
for file in sorted(RISK_CLASSIFIER_DIR.glob("*.pkl")):
    print(f"   ✓ {file.name}")
print("=" * 70)

# Double check by loading and checking model timestamp
import joblib
from datetime import datetime

model_path = RISK_CLASSIFIER_DIR / "logistic_regression_model.pkl"
if model_path.exists():
    mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    print(f"\n⏰ Model file last modified: {mod_time.strftime('%Y-%m-%d %I:%M:%S %p')}")
    
print("\n" + "=" * 70)
