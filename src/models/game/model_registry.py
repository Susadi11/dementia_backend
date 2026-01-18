# src/models/game/model_registry.py
"""
Model Registry: Load all ML models once at startup
Prevents slow API responses from reloading models per request
"""
import os
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Global Model Storage
# ============================================================================
_MODELS = {
    "lstm_model": None,
    "risk_classifier": None,
    "scaler": None,
    "lstm_scaler": None,
    "label_encoder": None
}

_MODEL_LOADED = False

# ============================================================================
# Path Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent.parent  # dementia_backend/
LSTM_MODEL_DIR = BASE_DIR / "src" / "models" / "game" / "lstm_model"
RISK_CLASSIFIER_DIR = BASE_DIR / "src" / "models" / "game" / "risk_classifier"

# ============================================================================
# LSTM Model Loading
# ============================================================================
def load_lstm_model():
    """
    Load LSTM model for temporal trend analysis.
    Expected file: src/models/game/lstm_model/lstm_model.keras (or .h5)
    """
    try:
        from tensorflow import keras
        
        # Try .keras first (TF 2.13+), fallback to .h5
        keras_path = LSTM_MODEL_DIR / "lstm_model.keras"
        h5_path = LSTM_MODEL_DIR / "lstm_model.h5"
        
        if keras_path.exists():
            model = keras.models.load_model(str(keras_path))
            logger.info(f"[OK] LSTM model loaded from {keras_path}")
        elif h5_path.exists():
            model = keras.models.load_model(str(h5_path))
            logger.info(f"[OK] LSTM model loaded from {h5_path}")
        else:
            logger.warning("[WARNING] LSTM model file not found, using dummy model")
            return None
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Failed to load LSTM model: {e}")
        return None

def load_lstm_scaler():
    """
    Load scaler for LSTM input normalization.
    Expected file: src/models/game/lstm_model/lstm_scaler.pkl
    """
    try:
        scaler_path = LSTM_MODEL_DIR / "lstm_scaler.pkl"
        
        if not scaler_path.exists():
            logger.warning("[WARNING] LSTM scaler not found, will skip scaling")
            return None
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        logger.info(f"[OK] LSTM scaler loaded from {scaler_path}")
        return scaler
        
    except Exception as e:
        logger.error(f"✗ Failed to load LSTM scaler: {e}")
        return None

# ============================================================================
# Risk Classifier Loading
# ============================================================================
def load_risk_classifier():
    """
    Load Logistic Regression risk classifier.
    Expected file: src/models/game/risk_classifier/logistic_regression_model.pkl
    """
    try:
        model_path = RISK_CLASSIFIER_DIR / "logistic_regression_model.pkl"
        
        if not model_path.exists():
            logger.warning("[WARNING] Risk classifier not found, using dummy classifier")
            return None
        
        model = joblib.load(model_path)
        
        logger.info(f"[OK] Risk classifier loaded from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"✗ Failed to load risk classifier: {e}")
        return None

def load_risk_scaler():
    """
    Load scaler for risk classifier input features.
    Expected file: src/models/game/risk_classifier/feature_scaler.pkl
    """
    try:
        scaler_path = RISK_CLASSIFIER_DIR / "feature_scaler.pkl"
        
        if not scaler_path.exists():
            logger.warning("[WARNING] Risk scaler not found, will skip scaling")
            return None
        
        scaler = joblib.load(scaler_path)
        
        logger.info(f"[OK] Risk scaler loaded from {scaler_path}")
        return scaler
        
    except Exception as e:
        logger.error(f"✗ Failed to load risk scaler: {e}")
        return None

def load_label_encoder():
    """
    Load label encoder for risk classifier output labels.
    Expected file: src/models/game/risk_classifier/label_encoder.pkl
    """
    try:
        encoder_path = RISK_CLASSIFIER_DIR / "label_encoder.pkl"
        
        if not encoder_path.exists():
            logger.warning("⚠ Label encoder not found, using default labels")
            return None
        
        encoder = joblib.load(encoder_path)
        
        logger.info(f"✓ Label encoder loaded from {encoder_path}")
        return encoder
        
    except Exception as e:
        logger.error(f"✗ Failed to load label encoder: {e}")
        return None

# ============================================================================
# Load All Models (Call at Startup)
# ============================================================================
def load_all_models():
    """
    Load all models into memory.
    Call this once at FastAPI startup.
    """
    global _MODEL_LOADED
    
    if _MODEL_LOADED:
        logger.info("Models already loaded, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("🔄 LOADING ML MODELS...")
    logger.info("=" * 60)
    
    _MODELS["lstm_model"] = load_lstm_model()
    _MODELS["lstm_scaler"] = load_lstm_scaler()
    _MODELS["risk_classifier"] = load_risk_classifier()
    _MODELS["scaler"] = load_risk_scaler()
    _MODELS["label_encoder"] = load_label_encoder()
    
    _MODEL_LOADED = True
    
    # Log detailed summary
    logger.info("=" * 60)
    logger.info("MODEL LOADING SUMMARY:")
    logger.info("-" * 60)
    logger.info(f"  LSTM Model: {'[OK] Loaded' if _MODELS['lstm_model'] is not None else '[WARN] Failed (will use dummy)'}")
    logger.info(f"  Risk Classifier: {'[OK] Loaded (' + _MODELS['risk_classifier'].__class__.__name__ + ')' if _MODELS['risk_classifier'] is not None else '[ERROR] FAILED - WILL USE RANDOM!'}")
    logger.info(f"  Feature Scaler: {'[OK] Loaded' if _MODELS['scaler'] is not None else '[WARN] Failed'}")
    logger.info(f"  Label Encoder: {'[OK] Loaded (classes: ' + str(_MODELS['label_encoder'].classes_ if _MODELS['label_encoder'] else 'None') + ')' if _MODELS['label_encoder'] is not None else '[WARN] Failed'}")
    logger.info("-" * 60)

    if _MODELS['risk_classifier'] is None:
        logger.error("[CRITICAL] Risk classifier FAILED to load!")
        logger.error("Check model file: src/models/game/risk_classifier/logistic_regression_model.pkl")
        logger.error("API will use RANDOM predictions until fixed!")
    else:
        logger.info(f"[OK] Risk classifier ready: {_MODELS['risk_classifier'].__class__.__name__}")

    logger.info("=" * 60)

# ============================================================================
# Getter Functions (Use in Services)
# ============================================================================
def get_lstm_model():
    """Get loaded LSTM model"""
    if not _MODEL_LOADED:
        load_all_models()
    return _MODELS["lstm_model"]

def get_lstm_scaler():
    """Get loaded LSTM scaler"""
    if not _MODEL_LOADED:
        load_all_models()
    return _MODELS["lstm_scaler"]

def get_risk_classifier():
    """Get loaded risk classifier"""
    if not _MODEL_LOADED:
        load_all_models()
    return _MODELS["risk_classifier"]

def get_risk_scaler():
    """Get loaded risk scaler"""
    if not _MODEL_LOADED:
        load_all_models()
    return _MODELS["scaler"]

def get_label_encoder():
    """Get loaded label encoder"""
    if not _MODEL_LOADED:
        load_all_models()
    return _MODELS["label_encoder"]

# ============================================================================
# Dummy Models (Fallback for Testing Without Trained Models)
# ============================================================================
class DummyLSTM:
    """Dummy LSTM that returns random decline score"""
    def predict(self, X):
        return np.random.rand(X.shape[0], 1) * 0.3  # 0-0.3 decline score

class DummyRiskClassifier:
    """Dummy classifier that returns random risk"""
    def predict_proba(self, X):
        # Returns [prob_low, prob_medium, prob_high]
        probs = np.random.dirichlet([2, 2, 1], size=X.shape[0])
        return probs
    
    def predict(self, X):
        return np.random.choice([0, 1, 2], size=X.shape[0])  # 0=LOW, 1=MED, 2=HIGH

def get_lstm_model_safe():
    """Get LSTM model, fallback to dummy if not loaded"""
    model = get_lstm_model()
    if model is None:
        logger.warning("Using dummy LSTM model")
        return DummyLSTM()
    return model

def get_risk_classifier_safe():
    """Get risk classifier, fallback to dummy if not loaded"""
    model = get_risk_classifier()
    if model is None:
        logger.warning("Using dummy risk classifier")
        return DummyRiskClassifier()
    return model