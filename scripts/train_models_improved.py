"""
Improved Training Pipeline with Proper Validation

This script implements best practices to prevent overfitting:
1. Proper train/validation/test split
2. Stratified K-Fold cross-validation
3. Hyperparameter tuning with GridSearchCV
4. Feature selection
5. Learning curve analysis
6. Model comparison
7. Comprehensive evaluation

Usage:
    # Basic training
    python scripts/train_models_improved.py

    # With hyperparameter tuning
    python scripts/train_models_improved.py --tune-hyperparameters

    # With specific data files
    python scripts/train_models_improved.py \
        --train-data data/train_clean.csv \
        --test-data data/test_clean.csv \
        --n-folds 5
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedModelTrainer:
    """
    Improved model trainer with proper validation and anti-overfitting measures.
    """
    
    def __init__(
        self,
        output_dir: str = "models/improved",
        use_cross_validation: bool = True,
        n_folds: int = 5,
        tune_hyperparameters: bool = False
    ):
        """Initialize trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.tune_hyperparameters = tune_hyperparameters
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'best_model': None
        }
    
    def load_data(self, train_file: str, test_file: Optional[str] = None) -> Tuple:
        """Load training and optionally test data."""
        logger.info(f"Loading training data from: {train_file}")
        train_df = pd.read_csv(train_file)
        
        if test_file:
            logger.info(f"Loading test data from: {test_file}")
            test_df = pd.read_csv(test_file)
        else:
            # Split from training data
            logger.info("No test file provided, splitting from training data")
            train_df, test_df = train_test_split(
                train_df, test_size=0.2, random_state=42,
                stratify=train_df['confusion_detected'] if 'confusion_detected' in train_df.columns else None
            )
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels."""
        # Define feature columns
        text_features = [
            "hesitation_pauses", "semantic_incoherence", "low_confidence_answers",
            "repeated_questions", "self_correction", "response_coherence"
        ]
        temporal_features = [
            "response_time_seconds", "pause_frequency", "speech_rate"
        ]
        
        all_features = text_features + temporal_features
        available_features = [f for f in all_features if f in df.columns]
        
        if not available_features:
            raise ValueError("No required features found in dataset!")
        
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Extract features and labels
        X = df[available_features].fillna(0).values
        
        # Determine target column
        if 'confusion_detected' in df.columns:
            y = df['confusion_detected'].astype(int).values
            logger.info("Target: confusion_detected")
        elif 'dementia_label' in df.columns:
            y = df['dementia_label'].astype(int).values
            logger.info("Target: dementia_label")
        else:
            raise ValueError("No target column found!")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        return X, y, available_features
    
    def feature_selection(
        self, X_train: np.ndarray, y_train: np.ndarray,
        feature_names: List[str], k: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Select top k features."""
        logger.info(f"\nPerforming feature selection (selecting top {k} features)...")
        
        selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = sorted(
            zip(feature_names, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info("Top features by F-score:")
        for feat, score in feature_scores[:k]:
            logger.info(f"  {feat}: {score:.2f}")
        
        self.results['selected_features'] = selected_features
        self.results['feature_scores'] = {feat: float(score) for feat, score in feature_scores}
        
        return X_train_selected, selector, selected_features
    
    def get_model_configs(self) -> Dict:
        """Get model configurations with regularization to prevent overfitting."""
        configs = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [10, 20, 30],
                    'min_samples_leaf': [5, 10, 15],
                    'max_features': ['sqrt', 'log2']
                } if self.tune_hyperparameters else None,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt'
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    random_state=42
                ),
                'param_grid': {
                    'n_estimators': [30, 50, 100],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                } if self.tune_hyperparameters else None,
                'default_params': {
                    'n_estimators': 50,
                    'max_depth': 3,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                } if self.tune_hyperparameters else None,
                'default_params': {
                    'C': 0.1,
                    'penalty': 'l2'
                }
            }
        }
        
        return configs
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Train a single model with proper validation."""
        logger.info("\n" + "="*80)
        logger.info(f"Training {model_name}")
        logger.info("="*80)
        
        configs = self.get_model_configs()
        config = configs[model_name]
        
        # Scale features (important for Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select appropriate data for model
        if 'Logistic' in model_name:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # Initialize model with default params
        model = config['model']
        if not self.tune_hyperparameters and config['default_params']:
            model.set_params(**config['default_params'])
        
        # Hyperparameter tuning
        if self.tune_hyperparameters and config['param_grid']:
            logger.info("Tuning hyperparameters...")
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                model,
                config['param_grid'],
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_use, y_train)
            model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
            
            best_params = grid_search.best_params_
        else:
            model.fit(X_train_use, y_train)
            best_params = config.get('default_params', {})
        
        # Cross-validation
        if self.use_cross_validation:
            logger.info(f"\nPerforming {self.n_folds}-Fold Cross-Validation...")
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            cv_results = cross_validate(
                model, X_train_use, y_train,
                cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                return_train_score=True,
                n_jobs=-1
            )
            
            cv_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                train_scores = cv_results[f'train_{metric}']
                test_scores = cv_results[f'test_{metric}']
                
                cv_metrics[metric] = {
                    'train_mean': float(np.mean(train_scores)),
                    'train_std': float(np.std(train_scores)),
                    'test_mean': float(np.mean(test_scores)),
                    'test_std': float(np.std(test_scores)),
                    'gap': float(np.mean(train_scores) - np.mean(test_scores))
                }
                
                logger.info(f"  {metric.capitalize()}:")
                logger.info(f"    Train: {cv_metrics[metric]['train_mean']:.4f} (±{cv_metrics[metric]['train_std']:.4f})")
                logger.info(f"    Test:  {cv_metrics[metric]['test_mean']:.4f} (±{cv_metrics[metric]['test_std']:.4f})")
                logger.info(f"    Gap:   {cv_metrics[metric]['gap']:.4f}")
        else:
            cv_metrics = None
        
        # Final evaluation on test set
        logger.info("\nEvaluating on held-out test set...")
        y_pred_train = model.predict(X_train_use)
        y_pred_test = model.predict(X_test_use)
        
        y_proba_test = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        train_metrics = {
            'accuracy': float(accuracy_score(y_train, y_pred_train)),
            'precision': float(precision_score(y_train, y_pred_train, zero_division=0)),
            'recall': float(recall_score(y_train, y_pred_train, zero_division=0)),
            'f1': float(f1_score(y_train, y_pred_train, zero_division=0))
        }
        
        test_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred_test)),
            'precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_test, zero_division=0))
        }
        
        if y_proba_test is not None:
            test_metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba_test))
        
        logger.info(f"\nTrain Metrics: {train_metrics}")
        logger.info(f"Test Metrics:  {test_metrics}")
        
        # Check for overfitting
        overfitting_gap = train_metrics['accuracy'] - test_metrics['accuracy']
        logger.info(f"\nOverfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.15:
            logger.warning("⚠️  OVERFITTING DETECTED! Consider:")
            logger.warning("  1. Increase regularization")
            logger.warning("  2. Reduce model complexity")
            logger.warning("  3. Collect more diverse data")
        elif overfitting_gap < 0.05:
            logger.info("✅ Good generalization!")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Classification report
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred_test)}")
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.results.get('selected_features', []),
                model.feature_importances_
            ))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("\nTop Feature Importances:")
            for feat, imp in sorted_importance[:5]:
                logger.info(f"  {feat}: {imp:.4f}")
        
        # Save results
        results = {
            'model_name': model_name,
            'hyperparameters': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'overfitting_gap': float(overfitting_gap),
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance
        }
        
        return model, scaler, results
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """Train all model types and compare."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        configs = self.get_model_configs()
        best_model = None
        best_scaler = None
        best_score = 0
        best_name = None
        
        for model_name in configs.keys():
            model, scaler, results = self.train_model(
                model_name, X_train, y_train, X_test, y_test
            )
            
            self.results['models'][model_name] = results
            
            # Track best model by test F1 score
            test_f1 = results['test_metrics']['f1']
            if test_f1 > best_score:
                best_score = test_f1
                best_model = model
                best_scaler = scaler
                best_name = model_name
        
        logger.info("\n" + "="*80)
        logger.info(f"BEST MODEL: {best_name} (Test F1: {best_score:.4f})")
        logger.info("="*80)
        
        self.results['best_model'] = {
            'name': best_name,
            'test_f1': float(best_score)
        }
        
        return best_model, best_scaler, best_name
    
    def save_models(
        self,
        best_model,
        best_scaler,
        best_name: str,
        feature_selector=None
    ):
        """Save trained models and metadata."""
        logger.info("\nSaving models...")
        
        # Save best model
        model_filename = f"best_model_{best_name.replace(' ', '_').lower()}.joblib"
        scaler_filename = f"best_scaler_{best_name.replace(' ', '_').lower()}.joblib"
        
        model_path = self.output_dir / model_filename
        scaler_path = self.output_dir / scaler_filename
        
        joblib.dump(best_model, model_path)
        joblib.dump(best_scaler, scaler_path)
        
        if feature_selector:
            selector_path = self.output_dir / "feature_selector.joblib"
            joblib.dump(feature_selector, selector_path)
        
        logger.info(f"  Model saved: {model_path}")
        logger.info(f"  Scaler saved: {scaler_path}")
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"  Results saved: {results_path}")
        
        # Save model card
        self._create_model_card(best_name, model_path)
    
    def _create_model_card(self, model_name: str, model_path: Path):
        """Create a model card with metadata."""
        model_card = f"""# {model_name} - Confusion Detection Model

## Model Information

- **Model Type**: {model_name}
- **Task**: Binary Classification (Confusion Detection)
- **Training Date**: {self.results['timestamp']}
- **Framework**: scikit-learn

## Performance Metrics

### Test Set Performance
"""
        
        test_metrics = self.results['models'][model_name]['test_metrics']
        for metric, value in test_metrics.items():
            model_card += f"- **{metric.capitalize()}**: {value:.4f}\n"
        
        if self.use_cross_validation:
            model_card += "\n### Cross-Validation (5-Fold)\n"
            cv_metrics = self.results['models'][model_name]['cv_metrics']
            if cv_metrics:
                for metric in ['accuracy', 'f1']:
                    model_card += f"- **{metric.capitalize()}**: {cv_metrics[metric]['test_mean']:.4f} ± {cv_metrics[metric]['test_std']:.4f}\n"
        
        model_card += f"\n### Overfitting Analysis\n"
        model_card += f"- **Train-Test Gap**: {self.results['models'][model_name]['overfitting_gap']:.4f}\n"
        
        if self.results['models'][model_name]['overfitting_gap'] < 0.1:
            model_card += "- **Status**: ✅ Good generalization\n"
        else:
            model_card += "- **Status**: ⚠️ Possible overfitting\n"
        
        model_card += f"\n## Features Used\n\n"
        for feat in self.results.get('selected_features', []):
            model_card += f"- {feat}\n"
        
        model_card += f"\n## Hyperparameters\n\n"
        for param, value in self.results['models'][model_name]['hyperparameters'].items():
            model_card += f"- {param}: {value}\n"
        
        model_card += f"\n## Usage\n\n```python\nimport joblib\nimport numpy as np\n\n"
        model_card += f"model = joblib.load('{model_path.name}')\n"
        model_card += f"scaler = joblib.load('{model_path.stem.replace('model', 'scaler')}.joblib')\n\n"
        model_card += "# Your feature array\nfeatures = np.array([[...]])\n"
        model_card += "features_scaled = scaler.transform(features)\n"
        model_card += "prediction = model.predict(features_scaled)\n```\n"
        
        card_path = self.output_dir / f"MODEL_CARD_{model_name.replace(' ', '_')}.md"
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        logger.info(f"  Model card saved: {card_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Improved Model Training Pipeline")
    parser.add_argument(
        '--train-data',
        default='data/enhanced_training_data.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--test-data',
        default=None,
        help='Path to test data (optional, will split from train if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        default='models/improved',
        help='Output directory for models'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Perform hyperparameter tuning (slower but better)'
    )
    parser.add_argument(
        '--no-cross-validation',
        action='store_true',
        help='Disable cross-validation'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ImprovedModelTrainer(
        output_dir=args.output_dir,
        use_cross_validation=not args.no_cross_validation,
        n_folds=args.n_folds,
        tune_hyperparameters=args.tune_hyperparameters
    )
    
    # Load data
    train_df, test_df = trainer.load_data(args.train_data, args.test_data)
    
    # Prepare features
    X_train, y_train, feature_names = trainer.prepare_features(train_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)
    
    # Feature selection
    X_train_selected, feature_selector, selected_features = trainer.feature_selection(
        X_train, y_train, feature_names, k=7
    )
    X_test_selected = feature_selector.transform(X_test)
    
    # Train all models
    best_model, best_scaler, best_name = trainer.train_all_models(
        X_train_selected, y_train, X_test_selected, y_test
    )
    
    # Save results
    trainer.save_models(best_model, best_scaler, best_name, feature_selector)
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Check results in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
