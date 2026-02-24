# Random Forest - Confusion Detection Model

## Model Information

- **Model Type**: Random Forest
- **Task**: Binary Classification (Confusion Detection)
- **Training Date**: 2026-02-22T00:03:56.077230
- **Framework**: scikit-learn

## Performance Metrics

### Test Set Performance
- **Accuracy**: 0.9615
- **Precision**: 0.9822
- **Recall**: 0.8973
- **F1**: 0.9379
- **Roc_auc**: 0.9928

### Cross-Validation (5-Fold)
- **Accuracy**: 0.9453 ± 0.0145
- **F1**: 0.9121 ± 0.0211

### Overfitting Analysis
- **Train-Test Gap**: -0.0071
- **Status**: ✅ Good generalization

## Features Used

- hesitation_pauses
- semantic_incoherence
- self_correction
- response_coherence
- response_time_seconds
- pause_frequency
- speech_rate

## Hyperparameters

- n_estimators: 100
- max_depth: 5
- min_samples_split: 20
- min_samples_leaf: 10
- max_features: sqrt

## Usage

```python
import joblib
import numpy as np

model = joblib.load('best_model_random_forest.joblib')
scaler = joblib.load('best_scaler_random_forest.joblib')

# Your feature array
features = np.array([[...]])
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
```
