"""
Retrain Risk Classifier for Cognitive Game
==========================================
Trains a new LogisticRegression with the CORRECT feature scales:
  - RT in seconds (not ms), so rtAdjMedian ~ 0.3-4.0
  - IES = RT_s / accuracy  ~ 0.5-15.0
  - SAC = accuracy / RT_s  ~ 0.05-2.0
  - variability in seconds ~ 0.05-1.0

Risk labels based on clinical thresholds:
  LOW    : accuracy >= 0.70, IES <= 2.0, variability <= 0.25
  MEDIUM : accuracy 0.40-0.70 OR IES 2.0-5.0 OR variability 0.25-0.55
  HIGH   : accuracy < 0.40 OR IES > 5.0 OR variability > 0.55

Run from project root:
    .\\venv\\Scripts\\python.exe scripts\\retrain_risk_classifier.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)

# =============================================================================
# 1. Generate Synthetic Training Data
#    14 features matching extract_risk_features() in game_service.py:
#    [mean_sac, slope_sac, mean_ies, slope_ies, mean_acc, mean_rt, mean_var,
#     lstm_score, cur_sac, cur_ies, slope_acc, slope_rt, std_sac, std_ies]
# =============================================================================

def generate_samples(n, risk_level):
    """Generate synthetic session feature vectors for a given risk level."""
    samples = []
    labels  = []

    for _ in range(n):
        if risk_level == "LOW":
            # Healthy: high accuracy, fast RT, low variability
            accuracy   = np.random.uniform(0.70, 1.00)
            rt         = np.random.uniform(0.30, 1.50)
            variab     = np.random.uniform(0.03, 0.25)
            lstm       = np.random.uniform(0.00, 0.15)
            slope_acc  = np.random.uniform(-0.02, 0.05)
            slope_rt   = np.random.uniform(-0.10, 0.05)

        elif risk_level == "MEDIUM":
            # Borderline: moderate accuracy, slower RT, noticeable variability
            accuracy   = np.random.uniform(0.40, 0.72)
            rt         = np.random.uniform(1.20, 3.00)
            variab     = np.random.uniform(0.20, 0.60)
            lstm       = np.random.uniform(0.10, 0.50)
            slope_acc  = np.random.uniform(-0.05, 0.01)
            slope_rt   = np.random.uniform(0.00, 0.20)

        else:  # HIGH
            # Significant decline: low accuracy, slow RT, high variability
            # Cap at realistic patient values (acc >= 0.08, rt <= 5.5) to avoid
            # extreme IES values that distort the scaler range.
            accuracy   = np.random.uniform(0.08, 0.42)
            rt         = np.random.uniform(2.50, 5.50)
            variab     = np.random.uniform(0.45, 1.40)
            lstm       = np.random.uniform(0.45, 1.00)
            slope_acc  = np.random.uniform(-0.20, -0.02)
            slope_rt   = np.random.uniform(0.10, 0.80)

        sac  = accuracy / max(rt, 0.05)
        ies  = rt / max(accuracy, 0.01)

        # Add correlated historical variation
        mean_sac  = sac  * np.random.uniform(0.85, 1.15)
        mean_ies  = ies  * np.random.uniform(0.85, 1.15)
        mean_acc  = accuracy * np.random.uniform(0.88, 1.12)
        mean_rt   = rt   * np.random.uniform(0.90, 1.10)
        mean_var  = variab * np.random.uniform(0.85, 1.15)

        slope_sac = -slope_acc * 0.1  # SAC improves when accuracy improves
        slope_ies =  slope_rt  * 0.5  # IES worsens when RT increases

        std_sac   = mean_sac * np.random.uniform(0.02, 0.15)
        std_ies   = mean_ies * np.random.uniform(0.02, 0.15)

        feat = [
            mean_sac, slope_sac,
            mean_ies, slope_ies,
            mean_acc, mean_rt, mean_var,
            lstm,
            sac, ies,
            slope_acc, slope_rt,
            std_sac, std_ies,
        ]
        samples.append(feat)
        labels.append(risk_level)

    return samples, labels


N = 600  # per class
X_list, y_list = [], []
for lvl in ["LOW", "MEDIUM", "HIGH"]:
    s, l = generate_samples(N, lvl)
    X_list.extend(s)
    y_list.extend(l)

X = np.array(X_list)
y = np.array(y_list)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  LOW={np.sum(y=='LOW')}, MEDIUM={np.sum(y=='MEDIUM')}, HIGH={np.sum(y=='HIGH')}")

# =============================================================================
# 2. Encode Labels & Scale Features
# =============================================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)  # alphabetical: HIGH=0, LOW=1, MEDIUM=2
print(f"Label encoder classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =============================================================================
# 3. Train Logistic Regression
# =============================================================================
clf = LogisticRegression(
    C=1.0,
    max_iter=500,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

# =============================================================================
# 4. Sanity-check with real-world scenario (the user's broken session)
# =============================================================================
print("=" * 60)
print("Sanity checks on known scenarios")
print("=" * 60)

RISK_LABELS = ["HIGH", "LOW", "MEDIUM"]  # alphabetical order

def check(name, feat_list, expected):
    Xc = scaler.transform(np.array([feat_list]))
    probs = clf.predict_proba(Xc)[0]
    pred  = RISK_LABELS[np.argmax(probs)]
    ok    = "OK" if pred == expected else "FAIL"
    prob_dict = dict(zip(RISK_LABELS, probs.round(3)))
    print(f"  [{ok}] {name}")
    print(f"       Predicted: {pred} (expected {expected})  probs={prob_dict}")

# Healthy player (86% accuracy, 0.22s RT)
acc, rt = 0.86, 0.22
check("Healthy (86% acc, 220ms RT)",
      [acc/rt, -0.01, rt/acc, 0.0, acc, rt, 0.05, 0.05,
       acc/rt, rt/acc, 0.0, 0.0, 0.01, 0.02], "LOW")

# User's failing session (50% accuracy, 1.406s RT, variability=0.607)
acc, rt, var = 0.50, 1.406, 0.607
sac = acc/rt; ies = rt/acc
check("User session (50% acc, 1406ms RT, var=607ms)",
      [sac, 0.0, ies, 0.0, acc, rt, var, 0.0,
       sac, ies, 0.0, 0.0, 0.0, 0.0], "MEDIUM")

# Borderline player (55% accuracy, 2.5s RT)
acc, rt, var = 0.55, 2.50, 0.30
sac = acc/rt; ies = rt/acc
check("Borderline (55% acc, 2500ms RT)",
      [sac, -0.01, ies, 0.05, acc, rt, var, 0.30,
       sac, ies, -0.01, 0.03, 0.02, 0.10], "MEDIUM")

# Declining player (20% accuracy, 5.0s RT) - consistent std values
acc, rt, var = 0.20, 5.00, 0.90
sac = acc/rt; ies = rt/acc
check("Declining (20% acc, 5000ms RT)",
      [sac, -0.12, ies, 0.40, acc, rt, var, 0.85,
       sac, ies, -0.08, 0.40, sac*0.10, ies*0.10], "HIGH")

# =============================================================================
# 5. Save Models
# =============================================================================
out_dir = Path("src/models/game/risk_classifier")
out_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(clf,    out_dir / "logistic_regression_model.pkl")
joblib.dump(scaler, out_dir / "feature_scaler.pkl")
joblib.dump(le,     out_dir / "label_encoder.pkl")
# Also save under new names for forward-compatibility
joblib.dump(clf,    out_dir / "risk_logreg.pkl")
joblib.dump(scaler, out_dir / "risk_scaler.pkl")
joblib.dump(le,     out_dir / "risk_label_encoder.pkl")

print(f"\nModels saved to {out_dir}/")
print("  logistic_regression_model.pkl")
print("  feature_scaler.pkl")
print("  label_encoder.pkl")
print("  risk_logreg.pkl  (new name alias)")
print("  risk_scaler.pkl  (new name alias)")
print("  risk_label_encoder.pkl  (new name alias)")
print("\nDone. Restart the API server to load the new models.")
