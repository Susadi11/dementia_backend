"""
Model Loader - Centralized Model Loading from Registry

Loads ML models from models_registry.json supporting:
- Hugging Face models (public)
- Local file models (.pkl, .keras, .joblib)
- Multiple team members' models

Usage:
    from src.utils.model_loader import ModelLoader

    loader = ModelLoader()
    model = loader.load_model("dementia_bert_xgboost")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized model loader that reads from models_registry.json
    and loads models from Hugging Face or local paths.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize ModelLoader.

        Args:
            registry_path: Path to models_registry.json (optional)
        """
        # Default registry path
        if registry_path is None:
            base_dir = Path(__file__).parent.parent.parent  # dementia_backend/
            self.registry_path = base_dir / "models" / "models_registry.json"
        else:
            self.registry_path = Path(registry_path)

        # Load registry
        self.registry = self._load_registry()

        # Cache for loaded models (singleton pattern)
        self._model_cache: Dict[str, Any] = {}

        logger.info(f"ModelLoader initialized with registry: {self.registry_path}")

    def _load_registry(self) -> Dict:
        """Load models registry JSON file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            logger.info(f"✓ Loaded registry with {registry['total_models']} models")
            return registry
        except Exception as e:
            logger.error(f"✗ Failed to load registry: {e}")
            return {"models": [], "total_models": 0}

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get model information from registry.

        Args:
            model_id: Model ID (e.g., "dementia_bert_xgboost")

        Returns:
            Model info dict or None if not found
        """
        for model in self.registry.get("models", []):
            if model["id"] == model_id:
                return model

        logger.warning(f"Model not found in registry: {model_id}")
        return None

    def load_model(self, model_id: str, force_reload: bool = False) -> Any:
        """
        Load a model by ID from the registry.

        Args:
            model_id: Model ID from registry (e.g., "dementia_bert_xgboost")
            force_reload: Force reload even if cached

        Returns:
            Loaded model object
        """
        # Check cache first
        if not force_reload and model_id in self._model_cache:
            logger.info(f"[CACHE] Using cached model: {model_id}")
            return self._model_cache[model_id]

        # Get model info from registry
        model_info = self.get_model_info(model_id)

        if model_info is None:
            raise ValueError(f"Model not found in registry: {model_id}")

        # Load based on model_source
        model_source = model_info.get("model_source", "local")

        if model_source == "huggingface":
            model = self._load_from_huggingface(model_info)
        else:
            model = self._load_from_local(model_info)

        # Cache the model
        self._model_cache[model_id] = model

        return model

    def _load_from_huggingface(self, model_info: Dict) -> Any:
        """
        Load model from Hugging Face.

        Args:
            model_info: Model metadata from registry

        Returns:
            Loaded model
        """
        try:
            from huggingface_hub import hf_hub_download

            repo_id = model_info["file_path"]  # e.g., "susadi/dementia_bert_xgboost_model"
            model_id = model_info["id"]

            logger.info(f"Loading model from Hugging Face: {repo_id}")

            # Determine file type and load accordingly
            model_type = model_info.get("type", "").lower()

            # For pickle/joblib models (XGBoost, scikit-learn)
            if "xgboost" in model_type or "classifier" in model_type:
                import joblib

                # Download main model file
                model_filename = self._infer_model_filename(model_info)
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename
                )

                model = joblib.load(model_path)
                logger.info(f"✓ Loaded model from HF: {model_id}")

                return model

            # For transformer models (handled separately - see get_model_repo_id)
            else:
                logger.info(f"✓ Model repo ID ready: {repo_id}")
                return {"repo_id": repo_id, "source": "huggingface"}

        except Exception as e:
            logger.error(f"✗ Failed to load model from Hugging Face: {e}")
            raise

    def _load_from_local(self, model_info: Dict) -> Any:
        """
        Load model from local file path.

        Args:
            model_info: Model metadata from registry

        Returns:
            Loaded model
        """
        try:
            file_path = Path(model_info["file_path"])

            # Make path absolute if relative
            if not file_path.is_absolute():
                base_dir = Path(__file__).parent.parent.parent
                file_path = base_dir / file_path

            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")

            logger.info(f"Loading local model: {file_path}")

            # Load based on file extension
            if file_path.suffix == '.keras' or file_path.suffix == '.h5':
                from tensorflow import keras
                model = keras.models.load_model(str(file_path))

            elif file_path.suffix == '.pkl':
                import pickle
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)

            elif file_path.suffix == '.joblib':
                import joblib
                model = joblib.load(file_path)

            else:
                raise ValueError(f"Unsupported model file type: {file_path.suffix}")

            logger.info(f"✓ Loaded local model: {model_info['id']}")
            return model

        except Exception as e:
            logger.error(f"✗ Failed to load local model: {e}")
            raise

    def _infer_model_filename(self, model_info: Dict) -> str:
        """
        Infer the model filename from model info.

        Args:
            model_info: Model metadata

        Returns:
            Likely model filename
        """
        model_type = model_info.get("type", "").lower()

        if "bert" in model_type and "xgboost" in model_type:
            return "dementia_xgboost_bert_model.pkl"
        elif "voice" in model_type and "xgboost" in model_type:
            return "dementia_voice_xgboost_model.pkl"
        else:
            # Default
            return "model.pkl"

    def get_model_repo_id(self, model_id: str) -> Optional[str]:
        """
        Get Hugging Face repo ID for a model (useful for transformer models).

        Args:
            model_id: Model ID from registry

        Returns:
            Hugging Face repo ID (e.g., "susadi/llama-3.2-3b-dementia-care")
        """
        model_info = self.get_model_info(model_id)

        if model_info is None:
            return None

        if model_info.get("model_source") == "huggingface":
            return model_info.get("file_path")

        # For transformer models with lora_adapter field
        if "lora_adapter" in model_info:
            return model_info["lora_adapter"]

        return None

    def list_models(self, category: Optional[str] = None) -> list:
        """
        List all available models.

        Args:
            category: Filter by category (e.g., "Conversational AI", "Game")

        Returns:
            List of model info dicts
        """
        models = self.registry.get("models", [])

        if category:
            models = [m for m in models if m.get("category") == category]

        return models


# Global singleton instance
_loader_instance: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Get global ModelLoader instance (singleton pattern).

    Returns:
        ModelLoader instance
    """
    global _loader_instance

    if _loader_instance is None:
        _loader_instance = ModelLoader()

    return _loader_instance
