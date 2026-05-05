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

# FIX SSL CERTIFICATE ISSUES (Must be BEFORE HuggingFace imports)
try:
    import certifi
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception:
    pass  # If certifi not available, continue without SSL override

logger = logging.getLogger(__name__)

# Global Model Storage
_MODELS = {
    "lstm_model": None,
    "risk_classifier": None,
    "scaler": None,
    "lstm_scaler": None,
    "label_encoder": None
}

_MODEL_LOADED = False

# Path Configuration
BASE_DIR = Path(__file__).parent.parent.parent.parent  # dementia_backend/
LSTM_MODEL_DIR = BASE_DIR / "src" / "models" / "game" / "lstm_model"
RISK_CLASSIFIER_DIR = BASE_DIR / "src" / "models" / "game" / "risk_classifier"

# Hugging Face Repository IDs
HF_LSTM_REPO = "vlakvindu/Dementia_LSTM_Model"
HF_RISK_REPO = "vlakvindu/Dementia_Risk_Clasification_model"


def _download_from_hf(repo_id: str, filename: str, local_dir: Path) -> Optional[Path]:
    """
    Download a single file from a public HuggingFace Hub repo into local_dir.
    Returns the local Path on success, or None on failure.
    
    SSL certificates are configured at module level to avoid system issues.
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


# LSTM Model Loading
def load_lstm_model():
    """
    Load LSTM model for temporal trend analysis.
    PRIORITY: Always download from HuggingFace (vlakvindu/Dementia_LSTM_Model)
    Caches to: src/models/game/lstm_model/.cache/huggingface/
    """
    try:
        from tensorflow import keras

        logger.info("[HuggingFace] Downloading LSTM model from vlakvindu/Dementia_LSTM_Model...")
        
        # Always download from HuggingFace (not from local files)
        keras_path = _download_from_hf(HF_LSTM_REPO, "lstm_model.keras", LSTM_MODEL_DIR)
        
        if keras_path is None or not keras_path.exists():
            logger.error("[CRITICAL] Failed to download LSTM model from HuggingFace")
            return None

        model = keras.models.load_model(str(keras_path))
        logger.info(f"✓ LSTM model loaded from HuggingFace: {keras_path}")
        return model

    except Exception as e:
        logger.error(f"✗ Failed to load LSTM model from HuggingFace: {e}")
        return None

def load_lstm_scaler():
    """
    Load scaler for LSTM input normalization.
    PRIORITY: Always download from HuggingFace (vlakvindu/Dementia_LSTM_Model)
    Caches to: src/models/game/lstm_model/.cache/huggingface/
    """
    try:
        logger.info("[HuggingFace] Downloading LSTM scaler from vlakvindu/Dementia_LSTM_Model...")
        
        # Always download from HuggingFace (not from local files)
        scaler_path = _download_from_hf(HF_LSTM_REPO, "lstm_scaler.pkl", LSTM_MODEL_DIR)
        
        if scaler_path is None or not scaler_path.exists():
            logger.error("[CRITICAL] Failed to download LSTM scaler from HuggingFace")
            return None

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        logger.info(f"✓ LSTM scaler loaded from HuggingFace: {scaler_path}")
        return scaler

    except Exception as e:
        logger.error(f"✗ Failed to load LSTM scaler from HuggingFace: {e}")
        return None


def load_risk_classifier():
    """
    Load Logistic Regression risk classifier.
    PRIORITY: Always download from HuggingFace (vlakvindu/Dementia_Risk_Clasification_model)
    Caches to: src/models/game/risk_classifier/.cache/huggingface/
    """
    try:
        logger.info("[HuggingFace] Downloading Risk Classifier from vlakvindu/Dementia_Risk_Clasification_model...")
        
        # Always download from HuggingFace (not from local files)
        model_path = _download_from_hf(HF_RISK_REPO, "risk_logreg.pkl", RISK_CLASSIFIER_DIR)
        
        if model_path is None or not model_path.exists():
            logger.error("[CRITICAL] Failed to download Risk Classifier from HuggingFace")
            return None

        model = joblib.load(model_path)
        logger.info(f"✓ Risk classifier loaded from HuggingFace: {model_path}")
        return model

    except Exception as e:
        logger.error(f"✗ Failed to load risk classifier from HuggingFace: {e}")
        return None

def load_risk_scaler():
    """
    Load scaler for risk classifier input features.
    PRIORITY: Always download from HuggingFace (vlakvindu/Dementia_Risk_Clasification_model)
    Caches to: src/models/game/risk_classifier/.cache/huggingface/
    """
    try:
        logger.info("[HuggingFace] Downloading Risk Scaler from vlakvindu/Dementia_Risk_Clasification_model...")
        
        # Always download from HuggingFace (not from local files)
        scaler_path = _download_from_hf(HF_RISK_REPO, "risk_scaler.pkl", RISK_CLASSIFIER_DIR)
        
        if scaler_path is None or not scaler_path.exists():
            logger.error("[CRITICAL] Failed to download Risk Scaler from HuggingFace")
            return None

        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Risk scaler loaded from HuggingFace: {scaler_path}")
        return scaler

    except Exception as e:
        logger.error(f"✗ Failed to load risk scaler from HuggingFace: {e}")
        return None

def load_label_encoder():
    """
    Load label encoder for risk classifier output labels.
    PRIORITY: Always download from HuggingFace (vlakvindu/Dementia_Risk_Clasification_model)
    Caches to: src/models/game/risk_classifier/.cache/huggingface/
    """
    try:
        logger.info("[HuggingFace] Downloading Label Encoder from vlakvindu/Dementia_Risk_Clasification_model...")
        
        # Always download from HuggingFace (not from local files)
        encoder_path = _download_from_hf(HF_RISK_REPO, "risk_label_encoder.pkl", RISK_CLASSIFIER_DIR)
        
        if encoder_path is None or not encoder_path.exists():
            logger.error("[CRITICAL] Failed to download Label Encoder from HuggingFace")
            return None

        encoder = joblib.load(encoder_path)
        logger.info(f"✓ Label encoder loaded from HuggingFace: {encoder_path}")
        return encoder

    except Exception as e:
        logger.error(f"✗ Failed to load label encoder from HuggingFace: {e}")
        return None

# Load All Models (Call at Startup)
def load_all_models():
    """
    Load all models from HuggingFace only (at startup).
    PRIORITY: Download from HuggingFace repositories
    - vlakvindu/Dementia_LSTM_Model
    - vlakvindu/Dementia_Risk_Clasification_model
    
    Caches models to local .cache/huggingface/ for reuse (fast on subsequent startups).
    Call this once at FastAPI startup.
    """
    global _MODEL_LOADED
    
    if _MODEL_LOADED:
        logger.info("Models already loaded, skipping")
        return
    
    logger.info("=" * 70)
    logger.info("🔄 LOADING ML MODELS FROM HUGGINGFACE ONLY")
    logger.info("=" * 70)
    logger.info("Priority: Download from uploaded HuggingFace repositories")
    logger.info("Cache: Local .cache/huggingface/ folders for faster reuse")
    logger.info("-" * 70)
    
    _MODELS["lstm_model"] = load_lstm_model()
    _MODELS["lstm_scaler"] = load_lstm_scaler()
    _MODELS["risk_classifier"] = load_risk_classifier()
    _MODELS["scaler"] = load_risk_scaler()
    _MODELS["label_encoder"] = load_label_encoder()
    
    _MODEL_LOADED = True
    
    # Log detailed summary
    logger.info("=" * 70)
    logger.info("MODEL LOADING SUMMARY:")
    logger.info("-" * 70)
    logger.info(f"  LSTM Model: {'✓ LOADED from HF' if _MODELS['lstm_model'] is not None else '✗ FAILED'}")
    logger.info(f"  LSTM Scaler: {'✓ LOADED from HF' if _MODELS['lstm_scaler'] is not None else '✗ FAILED'}")
    logger.info(f"  Risk Classifier: {'✓ LOADED from HF (' + _MODELS['risk_classifier'].__class__.__name__ + ')' if _MODELS['risk_classifier'] is not None else '✗ FAILED'}")
    logger.info(f"  Feature Scaler: {'✓ LOADED from HF' if _MODELS['scaler'] is not None else '✗ FAILED'}")
    logger.info(f"  Label Encoder: {'✓ LOADED from HF' if _MODELS['label_encoder'] is not None else '✗ FAILED'}")
    logger.info("-" * 70)

    # Check for critical failures
    if _MODELS['lstm_model'] is None:
        logger.error("[CRITICAL] LSTM model FAILED to load from HuggingFace!")
        logger.error("Repo: vlakvindu/Dementia_LSTM_Model")
    
    if _MODELS['risk_classifier'] is None:
        logger.error("[CRITICAL] Risk classifier FAILED to load from HuggingFace!")
        logger.error("Repo: vlakvindu/Dementia_Risk_Clasification_model")
    
    if _MODELS['lstm_model'] is not None and _MODELS['risk_classifier'] is not None:
        logger.info("✓ ALL MODELS LOADED SUCCESSFULLY FROM HUGGINGFACE!")
        logger.info("✓ Models cached to .cache/huggingface/ for fast reuse")
    else:
        logger.error("✗ CRITICAL: Some models failed to load from HuggingFace")

    logger.info("=" * 70)

# Getter Functions (Use in Services)
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

def get_lstm_model_safe():
    """Get LSTM model — requires trained model from HuggingFace.
    
    No fallback models for demo — using only your trained models.
    """
    model = get_lstm_model()
    if model is None:
        raise RuntimeError(
            "LSTM model failed to load from HuggingFace (vlakvindu/Dementia_LSTM_Model). "
            "Check your internet connection and try again. "
            "If the problem persists, verify the repository is public and hf_hub_download works."
        )
    return model

def get_risk_classifier_safe():
    """Get risk classifier — raises RuntimeError if model failed to load.

    Unlike the LSTM, the risk classifier is the *primary* output shown to the
    caregiver.  Returning a dummy / random prediction here would be medically
    misleading, so we intentionally fail with a clear 503 message instead.
    """
    model = get_risk_classifier()
    if model is None:
        raise RuntimeError(
            "Risk classifier failed to load from HuggingFace (vlakvindu/Dementia_Risk_Clasification_model). "
            "Check your internet connection and try again. "
            "If the problem persists, verify the repository is public and hf_hub_download works."
        )
    return model