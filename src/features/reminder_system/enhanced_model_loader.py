"""
Enhanced Model Loader for Reminder System

Loads the enhanced models trained with integrated Pitt Corpus data
and provides methods for prediction with the new feature set.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    """Loads and manages the enhanced reminder system models."""
    
    def __init__(self, models_dir: str = "models/reminder_system"):
        """
        Initialize the enhanced model loader.
        
        Args:
            models_dir: Directory containing the enhanced models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metadata = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all enhanced models from the models directory."""
        try:
            # Load models
            model_files = {
                'confusion_detection': 'confusion_detection_model.joblib',
                'cognitive_risk': 'cognitive_risk_model.joblib',
                'caregiver_alert': 'caregiver_alert_model.joblib',
                'response_classifier': 'response_classifier_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded enhanced {model_name} model")
                else:
                    logger.warning(f"❌ Enhanced {model_name} model not found at {model_path}")
            
            # Load scalers
            scaler_files = {
                'confusion_detection': 'confusion_detection_scaler.joblib',
                'cognitive_risk': 'cognitive_risk_scaler.joblib', 
                'caregiver_alert': 'caregiver_alert_scaler.joblib',
                'response_classifier': 'response_classifier_scaler.joblib'
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_path = self.models_dir / filename
                if scaler_path.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    logger.info(f"✅ Loaded enhanced {scaler_name} scaler")
            
            # Load encoder
            encoder_path = self.models_dir / 'response_classifier_encoder.joblib'
            if encoder_path.exists():
                self.encoders['response_classifier'] = joblib.load(encoder_path)
                logger.info(f"✅ Loaded enhanced response_classifier encoder")
            
            # Load metadata
            metadata_path = self.models_dir / 'enhanced_training_metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded enhanced training metadata")
                
            logger.info(f"🎉 Enhanced model system loaded successfully!")
            logger.info(f"📊 Models trained on {self.metadata.get('total_samples', 'unknown')} samples")
            
        except Exception as e:
            logger.error(f"Error loading enhanced models: {e}", exc_info=True)
            raise
    
    def predict_confusion_detection(self, features: pd.DataFrame) -> tuple:
        """Predict confusion detection using enhanced model."""
        if 'confusion_detection' not in self.models:
            raise ValueError("Enhanced confusion detection model not loaded")
        
        # Scale features if scaler available
        if 'confusion_detection' in self.scalers:
            features_scaled = self.scalers['confusion_detection'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = self.models['confusion_detection'].predict(features_scaled)[0]
        probability = self.models['confusion_detection'].predict_proba(features_scaled)[0]
        
        return prediction, max(probability)
    
    def predict_cognitive_risk(self, features: pd.DataFrame) -> tuple:
        """Predict cognitive risk using enhanced model."""
        if 'cognitive_risk' not in self.models:
            raise ValueError("Enhanced cognitive risk model not loaded")
        
        # Scale features if scaler available
        if 'cognitive_risk' in self.scalers:
            features_scaled = self.scalers['cognitive_risk'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = self.models['cognitive_risk'].predict(features_scaled)[0]
        probability = self.models['cognitive_risk'].predict_proba(features_scaled)[0]
        
        return prediction, max(probability)
    
    def predict_caregiver_alert(self, features: pd.DataFrame) -> tuple:
        """Predict caregiver alert need using enhanced model."""
        if 'caregiver_alert' not in self.models:
            raise ValueError("Enhanced caregiver alert model not loaded")
        
        # Scale features if scaler available
        if 'caregiver_alert' in self.scalers:
            features_scaled = self.scalers['caregiver_alert'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = self.models['caregiver_alert'].predict(features_scaled)[0]
        probability = self.models['caregiver_alert'].predict_proba(features_scaled)[0]
        
        return prediction, max(probability)
    
    def predict_response_classification(self, features: pd.DataFrame) -> tuple:
        """Classify response type using enhanced model."""
        if 'response_classifier' not in self.models:
            raise ValueError("Enhanced response classifier model not loaded")
        
        # Scale features if scaler available
        if 'response_classifier' in self.scalers:
            features_scaled = self.scalers['response_classifier'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = self.models['response_classifier'].predict(features_scaled)[0]
        probability = self.models['response_classifier'].predict_proba(features_scaled)[0]
        
        # Decode prediction if encoder available
        if 'response_classifier' in self.encoders:
            prediction_label = self.encoders['response_classifier'].inverse_transform([prediction])[0]
        else:
            prediction_label = prediction
        
        return prediction_label, max(probability)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'models_loaded': list(self.models.keys()),
            'scalers_loaded': list(self.scalers.keys()),
            'encoders_loaded': list(self.encoders.keys()),
            'metadata': self.metadata,
            'total_samples': self.metadata.get('total_samples', 'unknown') if self.metadata else 'unknown',
            'training_date': self.metadata.get('training_date', 'unknown') if self.metadata else 'unknown'
        }