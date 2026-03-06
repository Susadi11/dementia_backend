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
# Hugging Face Repository IDs
# ============================================================================
HF_LSTM_REPO = "vlakvindu/Dementia_LSTM_Model"
HF_RISK_REPO = "vlakvindu/Dementia_Risk_Clasification_model"


def _download_from_hf(repo_id: str, filename: str, local_dir: Path) -> Optional[Path]:
    """
    Download a single file from a public HuggingFace Hub repo into local_dir.
    Returns the local Path on success, or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {filename} from HuggingFace ({repo_id}) ...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir)
        )
        logger.info(f"[OK] Downloaded {filename} → {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download {filename} from {repo_id}: {e}")
        return None


# ============================================================================
# LSTM Model Loading
# ============================================================================
def load_lstm_model():
    """
    Load LSTM model for temporal trend analysis.
    Local path: src/models/game/lstm_model/lstm_model.keras
    HuggingFace fallback: vlakvindu/Dementia_LSTM_Model
    """
    try:
        from tensorflow import keras

        keras_path = LSTM_MODEL_DIR / "lstm_model.keras"
        h5_path = LSTM_MODEL_DIR / "lstm_model.h5"

        # Download from HuggingFace if not available locally
        if not keras_path.exists() and not h5_path.exists():
            keras_path = _download_from_hf(HF_LSTM_REPO, "lstm_model.keras", LSTM_MODEL_DIR)
            if keras_path is None:
                logger.warning("[WARNING] LSTM model unavailable locally and download failed")
                return None

        if keras_path and keras_path.exists():
            model = keras.models.load_model(str(keras_path))
            logger.info(f"[OK] LSTM model loaded from {keras_path}")
        elif h5_path.exists():
            model = keras.models.load_model(str(h5_path))
            logger.info(f"[OK] LSTM model loaded from {h5_path}")
        else:
            logger.warning("[WARNING] LSTM model file not found after download attempt")
            return None

        return model

    except Exception as e:
        logger.error(f"✗ Failed to load LSTM model: {e}")
        return None

def load_lstm_scaler():
    """
    Load scaler for LSTM input normalization.
    Local path: src/models/game/lstm_model/lstm_scaler.pkl
    HuggingFace fallback: vlakvindu/Dementia_LSTM_Model
    """
    try:
        scaler_path = LSTM_MODEL_DIR / "lstm_scaler.pkl"

        if not scaler_path.exists():
            downloaded = _download_from_hf(HF_LSTM_REPO, "lstm_scaler.pkl", LSTM_MODEL_DIR)
            if downloaded is None:
                logger.warning("[WARNING] LSTM scaler unavailable, will skip scaling")
                return None
            scaler_path = downloaded

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        logger.info(f"[OK] LSTM scaler loaded from {scaler_path}")
        return scaler

    except Exception as e:
        logger.error(f"✗ Failed to load LSTM scaler: {e}")
        return None


def load_risk_classifier():
    """
    Load Logistic Regression risk classifier.
    Local path: src/models/game/risk_classifier/risk_logreg.pkl
    HuggingFace fallback: vlakvindu/Dementia_Risk_Clasification_model
    """
    try:
        model_path = RISK_CLASSIFIER_DIR / "risk_logreg.pkl"
        if not model_path.exists():
            model_path = RISK_CLASSIFIER_DIR / "logistic_regression_model.pkl"

        if not model_path.exists():
            downloaded = _download_from_hf(HF_RISK_REPO, "risk_logreg.pkl", RISK_CLASSIFIER_DIR)
            if downloaded is None:
                logger.warning("[WARNING] Risk classifier unavailable, will use dummy")
                return None
            model_path = downloaded

        model = joblib.load(model_path)
        logger.info(f"[OK] Risk classifier loaded from {model_path}")
        return model

    except Exception as e:
        logger.error(f"✗ Failed to load risk classifier: {e}")
        return None

def load_risk_scaler():
    """
    Load scaler for risk classifier input features.
    Local path: src/models/game/risk_classifier/risk_scaler.pkl
    HuggingFace fallback: vlakvindu/Dementia_Risk_Clasification_model
    """
    try:
        scaler_path = RISK_CLASSIFIER_DIR / "risk_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = RISK_CLASSIFIER_DIR / "feature_scaler.pkl"

        if not scaler_path.exists():
            downloaded = _download_from_hf(HF_RISK_REPO, "risk_scaler.pkl", RISK_CLASSIFIER_DIR)
            if downloaded is None:
                logger.warning("[WARNING] Risk scaler unavailable, will skip scaling")
                return None
            scaler_path = downloaded

        scaler = joblib.load(scaler_path)
        logger.info(f"[OK] Risk scaler loaded from {scaler_path}")
        return scaler

    except Exception as e:
        logger.error(f"✗ Failed to load risk scaler: {e}")
        return None

def load_label_encoder():
    """
    Load label encoder for risk classifier output labels.
    Local path: src/models/game/risk_classifier/risk_label_encoder.pkl
    HuggingFace fallback: vlakvindu/Dementia_Risk_Clasification_model
    """
    try:
        encoder_path = RISK_CLASSIFIER_DIR / "risk_label_encoder.pkl"
        if not encoder_path.exists():
            encoder_path = RISK_CLASSIFIER_DIR / "label_encoder.pkl"

        if not encoder_path.exists():
            downloaded = _download_from_hf(HF_RISK_REPO, "risk_label_encoder.pkl", RISK_CLASSIFIER_DIR)
            if downloaded is None:
                logger.warning("[WARNING] Label encoder unavailable, using defaults")
                return None
            encoder_path = downloaded

        encoder = joblib.load(encoder_path)
        logger.info(f"[OK] Label encoder loaded from {encoder_path}")
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
        logger.error("Check model file: src/models/game/risk_classifier/risk_logreg.pkl")
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
    def predict(self, X, **kwargs):
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