"""
Context-Aware Reminder System Model Training

Trains models specifically for reminder system using:
1. Synthetic reminder-response data
2. Pitt Corpus dementia patterns
3. Combined text and voice features
4. Multi-task learning approach

This creates models optimized for reminder context analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.conversational_ai.feature_extractor import FeatureExtractor
from src.utils.helpers import calculate_overall_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReminderSystemTrainer:
    """
    Trains models specifically for context-aware reminder system.
    
    Creates specialized models for:
    1. Confusion detection in reminder responses
    2. Cognitive risk assessment
    3. Caregiver alert prediction
    4. Response type classification
    """
    
    def __init__(self, models_dir: str = "models/reminder_system"):
        """Initialize trainer with model directory."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Feature groups for different model types
        self.feature_groups = {
            "text_features": [
                "hesitation_pauses", "semantic_incoherence", "low_confidence_answers",
                "repeated_questions", "self_correction", "response_coherence"
            ],
            "temporal_features": [
                "response_time_seconds", "pause_frequency", "speech_rate"
            ],
            "cognitive_features": [
                "cognitive_risk_score", "confusion_detected", "memory_issue"
            ],
            "context_features": [
                "category_encoded", "priority_encoded", "time_of_day_encoded"
            ]
        }
    
    def load_training_data(self, data_file: str) -> pd.DataFrame:
        """Load and preprocess training data."""
        logger.info(f"Loading training data from {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples")
        
        # Basic preprocessing
        df = self._preprocess_data(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data preprocessing."""
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        text_columns = df.select_dtypes(include=[object]).columns
        df[text_columns] = df[text_columns].fillna("unknown")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Normalize response times (cap at reasonable maximum)
        if 'response_time_seconds' in df.columns:
            df['response_time_seconds'] = np.clip(df['response_time_seconds'], 0, 300)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for reminder system."""
        # Encode categorical variables
        categorical_cols = ['category', 'cognitive_level', 'data_source']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        # Time-based features
        if 'response_time_seconds' in df.columns:
            df['response_speed_category'] = pd.cut(
                df['response_time_seconds'], 
                bins=[0, 10, 30, 60, 300], 
                labels=['very_fast', 'fast', 'normal', 'slow']
            )
            df['response_speed_encoded'] = LabelEncoder().fit_transform(df['response_speed_category'].astype(str))
        
        # Cognitive severity levels
        if 'cognitive_risk_score' in df.columns:
            df['cognitive_severity'] = pd.cut(
                df['cognitive_risk_score'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['low', 'mild', 'moderate', 'high']
            )
            df['cognitive_severity_encoded'] = LabelEncoder().fit_transform(df['cognitive_severity'].astype(str))
        
        # Interaction complexity
        text_features = ['hesitation_pauses', 'semantic_incoherence', 'low_confidence_answers']
        if all(col in df.columns for col in text_features):
            df['interaction_complexity'] = (
                df['hesitation_pauses'] * 0.3 + 
                df['semantic_incoherence'] * 0.5 + 
                df['low_confidence_answers'] * 0.2
            )
        
        # Combined risk indicator
        risk_features = ['confusion_detected', 'memory_issue']
        if all(col in df.columns for col in risk_features):
            df['combined_risk'] = df['confusion_detected'].astype(int) + df['memory_issue'].astype(int)
        
        return df
    
    def train_confusion_detection_model(self, df: pd.DataFrame) -> Dict:
        """Train model to detect confusion in reminder responses."""
        logger.info("Training confusion detection model...")
        
        # Prepare features and target
        feature_cols = (
            self.feature_groups["text_features"] + 
            self.feature_groups["temporal_features"]
        )
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        y = df['confusion_detected'].astype(int)
        
        # Handle class imbalance
        class_weights = self._calculate_class_weights(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weights,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight=class_weights,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            score = roc_auc_score(y_test, model.predict_proba(X_test_scaled if name == 'logistic_regression' else X_test)[:, 1])
            model_results[name] = {
                'model': model,
                'score': score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Save best model
        self.models['confusion_detection'] = best_model
        self.scalers['confusion_detection'] = scaler if isinstance(best_model, LogisticRegression) else None
        
        # Save model
        model_path = self.models_dir / "confusion_detection_model.joblib"
        joblib.dump(best_model, model_path)
        
        if self.scalers['confusion_detection']:
            scaler_path = self.models_dir / "confusion_detection_scaler.joblib"
            joblib.dump(scaler, scaler_path)
        
        logger.info(f"Confusion detection model saved. Best score: {best_score:.3f}")
        
        return {
            'model_path': str(model_path),
            'best_score': best_score,
            'feature_importance': dict(zip(available_features, best_model.feature_importances_)) if hasattr(best_model, 'feature_importances_') else None,
            'results': model_results
        }
    
    def train_cognitive_risk_model(self, df: pd.DataFrame) -> Dict:
        """Train model to assess cognitive risk from responses."""
        logger.info("Training cognitive risk assessment model...")
        
        # Prepare features and target
        feature_cols = (
            self.feature_groups["text_features"] + 
            self.feature_groups["temporal_features"] +
            ["interaction_complexity", "combined_risk"]
        )
        
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        y = df['cognitive_risk_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train regression model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        self.models['cognitive_risk'] = model
        self.scalers['cognitive_risk'] = scaler
        
        model_path = self.models_dir / "cognitive_risk_model.joblib"
        scaler_path = self.models_dir / "cognitive_risk_scaler.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Cognitive risk model saved. R² score: {r2:.3f}, MSE: {mse:.3f}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'r2_score': r2,
            'mse': mse,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
    
    def train_caregiver_alert_model(self, df: pd.DataFrame) -> Dict:
        """Train model to predict when caregiver alerts are needed."""
        logger.info("Training caregiver alert prediction model...")
        
        # Prepare features and target
        feature_cols = (
            self.feature_groups["text_features"] + 
            self.feature_groups["temporal_features"] +
            self.feature_groups["cognitive_features"] +
            ["interaction_complexity", "combined_risk", "cognitive_severity_encoded"]
        )
        
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        y = df['caregiver_alert_needed'].astype(int)
        
        # Handle class imbalance
        class_weights = self._calculate_class_weights(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with cross-validation
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Final training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        self.models['caregiver_alert'] = model
        self.scalers['caregiver_alert'] = scaler
        
        model_path = self.models_dir / "caregiver_alert_model.joblib"
        scaler_path = self.models_dir / "caregiver_alert_scaler.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Caregiver alert model saved. AUC: {auc_score:.3f}, CV mean: {cv_scores.mean():.3f}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'auc_score': auc_score,
            'cv_scores': cv_scores.tolist(),
            'classification_report': report,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
    
    def train_response_classifier(self, df: pd.DataFrame) -> Dict:
        """Train model to classify response types."""
        logger.info("Training response type classifier...")
        
        if 'cognitive_level' not in df.columns:
            logger.warning("No cognitive_level column found, skipping response classifier")
            return {}
        
        # Prepare features and target
        feature_cols = (
            self.feature_groups["text_features"] + 
            self.feature_groups["temporal_features"] +
            ["interaction_complexity"]
        )
        
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        y = df['cognitive_level']
        
        # Encode target
        if 'cognitive_level' not in self.encoders:
            self.encoders['cognitive_level'] = LabelEncoder()
        
        y_encoded = self.encoders['cognitive_level'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        self.models['response_classifier'] = model
        
        model_path = self.models_dir / "response_classifier_model.joblib"
        encoder_path = self.models_dir / "response_classifier_encoder.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(self.encoders['cognitive_level'], encoder_path)
        
        logger.info(f"Response classifier saved. Accuracy: {report['accuracy']:.3f}")
        
        return {
            'model_path': str(model_path),
            'encoder_path': str(encoder_path),
            'accuracy': report['accuracy'],
            'classification_report': report,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
    
    def train_all_models(self, data_file: str) -> Dict:
        """Train all reminder system models."""
        logger.info("Starting comprehensive model training...")
        
        # Load data
        df = self.load_training_data(data_file)
        
        # Train all models
        results = {}
        
        try:
            results['confusion_detection'] = self.train_confusion_detection_model(df)
        except Exception as e:
            logger.error(f"Failed to train confusion detection model: {e}")
            results['confusion_detection'] = {'error': str(e)}
        
        try:
            results['cognitive_risk'] = self.train_cognitive_risk_model(df)
        except Exception as e:
            logger.error(f"Failed to train cognitive risk model: {e}")
            results['cognitive_risk'] = {'error': str(e)}
        
        try:
            results['caregiver_alert'] = self.train_caregiver_alert_model(df)
        except Exception as e:
            logger.error(f"Failed to train caregiver alert model: {e}")
            results['caregiver_alert'] = {'error': str(e)}
        
        try:
            results['response_classifier'] = self.train_response_classifier(df)
        except Exception as e:
            logger.error(f"Failed to train response classifier: {e}")
            results['response_classifier'] = {'error': str(e)}
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'data_file': data_file,
            'data_shape': df.shape,
            'feature_groups': self.feature_groups,
            'models_trained': list(results.keys()),
            'training_results': results
        }
        
        metadata_path = self.models_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training complete. Results saved to {self.models_dir}")
        return results
    
    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """Calculate class weights for imbalanced datasets."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))
    
    def load_trained_models(self):
        """Load all trained models for inference."""
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
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train context-aware reminder system models')
    parser.add_argument('--data', required=True, help='Path to combined training data CSV')
    parser.add_argument('--models-dir', default='models/reminder_system', help='Directory to save models')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ReminderSystemTrainer(args.models_dir)
    
    # Train all models
    results = trainer.train_all_models(args.data)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        if 'error' in result:
            print(f"  ❌ Failed: {result['error']}")
        else:
            if 'auc_score' in result:
                print(f"  ✅ AUC Score: {result['auc_score']:.3f}")
            if 'r2_score' in result:
                print(f"  ✅ R² Score: {result['r2_score']:.3f}")
            if 'accuracy' in result:
                print(f"  ✅ Accuracy: {result['accuracy']:.3f}")
    
    print(f"\nModels saved to: {args.models_dir}")


if __name__ == '__main__':
    main()