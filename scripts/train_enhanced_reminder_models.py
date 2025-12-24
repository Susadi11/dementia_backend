"""
Enhanced Context-Aware Reminder System Model Training

Integrates both synthetic reminder data and real Pitt Corpus dementia data
for comprehensive model training.

Features:
1. Processes Pitt Corpus .cha files for real-world dementia patterns
2. Combines synthetic reminder data with real dementia markers
3. Creates robust models trained on both synthetic and real data
4. Enhanced feature extraction from conversational patterns

Usage:
  python scripts/train_enhanced_reminder_models.py
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


class EnhancedReminderSystemTrainer:
    """
    Enhanced trainer that combines synthetic and Pitt Corpus data
    for more robust reminder system models.
    """
    
    def __init__(self, models_dir: str = "models/reminder_system"):
        """Initialize enhanced trainer."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Enhanced feature groups including Pitt-derived features
        self.feature_groups = {
            "text_features": [
                "hesitation_pauses", "semantic_incoherence", "low_confidence_answers",
                "repeated_questions", "self_correction", "response_coherence",
                "word_finding_difficulty", "circumlocution", "tangentiality"
            ],
            "temporal_features": [
                "response_time_seconds", "pause_frequency", "speech_rate",
                "utterance_length", "pause_duration"
            ],
            "cognitive_features": [
                "cognitive_risk_score", "confusion_detected", "memory_issue",
                "semantic_drift", "discourse_coherence", "lexical_diversity"
            ],
            "context_features": [
                "category_encoded", "priority_encoded", "time_of_day_encoded",
                "task_type_encoded", "dementia_severity"
            ],
            "pitt_features": [
                "pitt_dementia_markers", "narrative_coherence", "task_completion",
                "linguistic_complexity", "error_patterns"
            ]
        }
    
    def parse_cha_file(self, file_path: Path) -> str:
        """Extract participant utterances from CHAT .cha file."""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.strip()
                    if line.startswith('*PAR:'):
                        # Remove speaker tag and clean content
                        content = line.split('\t')[-1] if '\t' in line else line[5:]
                        # Remove timestamps and annotations
                        content = self._clean_cha_content(content)
                        if content.strip():
                            texts.append(content)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
        return ' '.join(texts)
    
    def _clean_cha_content(self, content: str) -> str:
        """Clean CHAT format annotations from content."""
        import re
        
        # Remove timestamps like 3754_5640
        content = re.sub(r'\d+_\d+', '', content)
        
        # Remove annotations like <text> [/] [//]
        content = re.sub(r'<[^>]*>', '', content)
        content = re.sub(r'\[/+\]', '', content)
        
        # Remove repetition markers and fillers
        content = re.sub(r'&-\w+', '', content)
        
        # Remove extra whitespace
        content = ' '.join(content.split())
        
        return content
    
    def load_pitt_corpus_data(self) -> pd.DataFrame:
        """Load and process Pitt Corpus data."""
        logger.info("Loading Pitt Corpus data...")
        
        pitt_dir = Path('data/Pitt')
        if not pitt_dir.exists():
            logger.warning('Pitt data not found, skipping...')
            return pd.DataFrame()
        
        pitt_data = []
        
        # Process Control and Dementia groups
        for group in ['Control', 'Dementia']:
            group_dir = pitt_dir / group
            if not group_dir.exists():
                continue
                
            dementia_label = 1 if group == 'Dementia' else 0
            
            # Process all tasks
            for task_dir in group_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                
                for cha_file in task_dir.glob('*.cha'):
                    try:
                        # Extract text
                        participant_text = self.parse_cha_file(cha_file)
                        if not participant_text.strip():
                            continue
                        
                        # Extract features
                        features = self._extract_pitt_features(participant_text, task_name)
                        
                        # Add metadata
                        features.update({
                            'participant_id': cha_file.stem,
                            'task_type': task_name,
                            'dementia_label': dementia_label,
                            'data_source': 'pitt_corpus',
                            'original_text': participant_text
                        })
                        
                        pitt_data.append(features)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {cha_file}: {e}")
        
        logger.info(f"Loaded {len(pitt_data)} Pitt Corpus samples")
        return pd.DataFrame(pitt_data)
    
    def _extract_pitt_features(self, text: str, task_type: str) -> Dict:
        """Extract features from Pitt Corpus text."""
        # Use existing feature extractor
        try:
            features = self.feature_extractor.extract_text_features(text)
        except:
            # Fallback basic features
            features = self._extract_basic_features(text)
        
        # Add Pitt-specific features
        features.update({
            'utterance_count': len(text.split('.')) if text else 0,
            'average_utterance_length': len(text.split()) / max(1, len(text.split('.'))) if text else 0,
            'task_type_encoded': self._encode_task_type(task_type),
            'linguistic_complexity': self._calculate_complexity(text),
            'narrative_coherence': self._assess_coherence(text, task_type),
        })
        
        return features
    
    def _extract_basic_features(self, text: str) -> Dict:
        """Extract basic features when advanced extraction fails."""
        words = text.split() if text else []
        sentences = text.split('.') if text else []
        
        return {
            'hesitation_pauses': text.count('&-uh') + text.count('um') + text.count('uh'),
            'semantic_incoherence': len([w for w in words if len(w) < 2]) / max(1, len(words)),
            'low_confidence_answers': text.count('maybe') + text.count('I think') + text.count('probably'),
            'repeated_questions': 0,  # Hard to detect without context
            'self_correction': text.count('[/]') + text.count('[//]'),
            'response_coherence': min(1.0, len(sentences) / max(1, len(words) / 10)),
            'word_finding_difficulty': text.count('...') + text.count('&-'),
            'circumlocution': 0,  # Requires semantic analysis
            'tangentiality': 0,  # Requires context analysis
            'response_time_seconds': np.random.uniform(5, 30),  # Placeholder
            'pause_frequency': text.count('...') + text.count('&-'),
            'speech_rate': len(words) / max(1, len(sentences)),
            'cognitive_risk_score': 0.5,  # Will be calculated later
            'confusion_detected': False,  # Will be determined by model
            'memory_issue': 'memory' in text.lower() or 'remember' in text.lower()
        }
    
    def _encode_task_type(self, task_type: str) -> int:
        """Encode task type to numerical value."""
        task_mapping = {
            'cookie': 0, 'fluency': 1, 'recall': 2, 'sentence': 3,
            'medication': 4, 'meal': 5, 'appointment': 6, 'hygiene': 7, 'safety': 8
        }
        return task_mapping.get(task_type.lower(), 9)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate linguistic complexity score."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = text.split('.')
        
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        avg_sentence_length = len(words) / max(1, len(sentences))
        unique_words = len(set(words)) / max(1, len(words))
        
        complexity = (avg_word_length / 10 + avg_sentence_length / 20 + unique_words) / 3
        return min(1.0, complexity)
    
    def _assess_coherence(self, text: str, task_type: str) -> float:
        """Assess narrative coherence based on task type."""
        if not text:
            return 0.0
        
        # Task-specific coherence assessment
        if task_type == 'cookie':
            # Look for typical cookie theft description elements
            elements = ['kitchen', 'boy', 'girl', 'cookie', 'jar', 'stool', 'water', 'dishes']
            found_elements = sum(1 for element in elements if element in text.lower())
            return found_elements / len(elements)
        
        # General coherence (simplified)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Check for logical flow (very basic)
        coherence_score = 0.7  # Default moderate coherence
        return coherence_score
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine synthetic and Pitt Corpus datasets."""
        logger.info("Combining datasets...")
        
        # Load synthetic data
        synthetic_df = pd.read_csv("data/synthetic_reminder_data.csv")
        synthetic_df['data_source'] = 'synthetic'
        logger.info(f"Loaded {len(synthetic_df)} synthetic samples")
        
        # Load Pitt Corpus data
        pitt_df = self.load_pitt_corpus_data()
        
        if pitt_df.empty:
            logger.warning("No Pitt data loaded, using synthetic data only")
            return self._prepare_synthetic_data(synthetic_df)
        
        # Align features between datasets
        combined_df = self._align_and_combine_datasets(synthetic_df, pitt_df)
        
        logger.info(f"Combined dataset size: {len(combined_df)} samples")
        return combined_df
    
    def _prepare_synthetic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare synthetic data with enhanced features."""
        # Add missing features with default values
        df['task_type_encoded'] = df.get('category_encoded', 0)
        df['dementia_severity'] = df.get('cognitive_risk_score', 0.5)
        df['pitt_dementia_markers'] = 0  # No Pitt markers in synthetic data
        df['narrative_coherence'] = 1.0 - df.get('semantic_incoherence', 0.0)
        df['task_completion'] = 1.0 - df.get('confusion_detected', 0.0).astype(float)
        df['linguistic_complexity'] = df.get('response_coherence', 0.5)
        df['error_patterns'] = df.get('self_correction', 0)
        
        return df
    
    def _align_and_combine_datasets(self, synthetic_df: pd.DataFrame, pitt_df: pd.DataFrame) -> pd.DataFrame:
        """Align features between synthetic and Pitt datasets."""
        # Get common features
        synthetic_features = set(synthetic_df.columns)
        pitt_features = set(pitt_df.columns)
        
        # Add missing features to both datasets
        for feature in synthetic_features - pitt_features:
            if feature not in ['reminder_id', 'reminder_text', 'user_response', 'timestamp']:
                pitt_df[feature] = self._get_default_value(feature, len(pitt_df))
        
        for feature in pitt_features - synthetic_features:
            if feature not in ['participant_id', 'original_text']:
                synthetic_df[feature] = self._get_default_value(feature, len(synthetic_df))
        
        # Map Pitt labels to synthetic format
        if 'dementia_label' in pitt_df.columns:
            # Create synthetic-style targets from Pitt data
            pitt_df['caregiver_alert_needed'] = pitt_df['dementia_label']
            pitt_df['cognitive_risk_score'] = pitt_df['dementia_label'] * 0.8 + np.random.uniform(0, 0.2, len(pitt_df))
            pitt_df['confusion_detected'] = pitt_df['dementia_label'].astype(bool)
            pitt_df['memory_issue'] = pitt_df['dementia_label'].astype(bool)
        
        # Combine datasets
        combined_df = pd.concat([synthetic_df, pitt_df], ignore_index=True, sort=False)
        
        return combined_df
    
    def _get_default_value(self, feature: str, size: int):
        """Get default value for missing features."""
        if 'encoded' in feature or 'score' in feature:
            return np.zeros(size)
        elif 'detected' in feature or feature.endswith('_issue'):
            return np.zeros(size, dtype=bool)
        elif 'time' in feature:
            return np.random.uniform(5, 60, size)
        else:
            return np.zeros(size)
    
    def train_enhanced_models(self, df: pd.DataFrame) -> Dict:
        """Train models with enhanced dataset."""
        logger.info("Training enhanced models...")
        
        # Prepare training data
        X, y_dict = self._prepare_training_data(df)
        
        results = {}
        
        # Train each model type
        for model_name, y in y_dict.items():
            logger.info(f"Training {model_name} model...")
            
            try:
                model_result = self._train_single_model(X, y, model_name)
                results[model_name] = model_result
                
                # Save model
                self._save_model_components(model_name, model_result)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare features and targets for training."""
        # Get feature columns
        feature_columns = []
        for group in self.feature_groups.values():
            feature_columns.extend([col for col in group if col in df.columns])
        
        # Remove duplicates
        feature_columns = list(set(feature_columns))
        
        # Handle missing features
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Prepare targets
        y_dict = {}
        
        # Confusion detection
        if 'confusion_detected' in df.columns:
            y_dict['confusion_detection'] = df['confusion_detected'].astype(int)
        elif 'dementia_label' in df.columns:
            y_dict['confusion_detection'] = df['dementia_label'].astype(int)
        else:
            y_dict['confusion_detection'] = pd.Series([0] * len(df))
        
        # Cognitive risk
        if 'cognitive_risk_score' in df.columns:
            y_dict['cognitive_risk'] = df['cognitive_risk_score']
        elif 'dementia_label' in df.columns:
            y_dict['cognitive_risk'] = df['dementia_label'].astype(float)
        else:
            y_dict['cognitive_risk'] = pd.Series([0.5] * len(df))
        
        # Caregiver alert
        if 'caregiver_alert_needed' in df.columns:
            y_dict['caregiver_alert'] = df['caregiver_alert_needed'].astype(int)
        elif 'dementia_label' in df.columns:
            y_dict['caregiver_alert'] = df['dementia_label'].astype(int)
        else:
            y_dict['caregiver_alert'] = pd.Series([0] * len(df))
        
        # Response classifier
        if 'category_encoded' in df.columns:
            y_dict['response_classifier'] = df['category_encoded']
        elif 'task_type_encoded' in df.columns:
            y_dict['response_classifier'] = df['task_type_encoded']
        else:
            # Create diverse categories
            y_dict['response_classifier'] = pd.Series(np.random.randint(0, 4, len(df)))
        
        return X, y_dict
    
    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """Train a single model with cross-validation."""
        # Check for valid data
        if len(y) == 0:
            raise ValueError(f"No data available for {model_name}")
        
        # Determine stratification
        stratify = None
        if len(np.unique(y)) > 1 and len(y) >= 10:
            # Check if each class has at least 2 samples
            unique_values, counts = np.unique(y, return_counts=True)
            if np.all(counts >= 2):
                stratify = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Determine problem type
        is_classification = len(np.unique(y)) < 10
        
        if is_classification:
            # Classification models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            # Regression models
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
        
        # Train and evaluate models
        best_model = None
        best_score = -np.inf
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation with proper handling
            try:
                cv_folds = min(5, len(X_train))  # Use fewer folds if needed
                if cv_folds < 2:
                    cv_folds = 2
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {e}")
                # Fallback to simple train score
                cv_scores = np.array([model.score(X_train_scaled, y_train)])
            
            # Test score
            test_score = model.score(X_test_scaled, y_test)
            
            results[name] = {
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'test_score': test_score
            }
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
        
        return {
            'best_model': best_model,
            'scaler': scaler,
            'best_score': best_score,
            'results': {k: {
                'cv_score_mean': float(v['cv_score_mean']),
                'cv_score_std': float(v['cv_score_std']),
                'test_score': float(v['test_score'])
            } for k, v in results.items()},
            'feature_columns': X.columns.tolist()
        }
    
    def _save_model_components(self, model_name: str, model_result: Dict):
        """Save model, scaler, and metadata."""
        model_path = self.models_dir / f"{model_name}_model.joblib"
        scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
        
        joblib.dump(model_result['best_model'], model_path)
        joblib.dump(model_result['scaler'], scaler_path)
        
        logger.info(f"Saved {model_name} model to {model_path}")
    
    def save_training_metadata(self, results: Dict, df: pd.DataFrame):
        """Save training metadata and results."""
        # Prepare serializable results
        serializable_results = {}
        for k, v in results.items():
            if 'error' in v:
                serializable_results[k] = {'error': v['error']}
            else:
                serializable_results[k] = {
                    'best_score': float(v['best_score']) if 'best_score' in v else 0.0,
                    'results': v.get('results', {}),
                    'feature_count': len(v.get('feature_columns', []))
                }
        
        metadata = {
            "training_date": datetime.now().isoformat(),
            "data_sources": ["synthetic", "pitt_corpus"],
            "synthetic_samples": len(df[df['data_source'] == 'synthetic']) if 'data_source' in df.columns else 0,
            "pitt_samples": len(df[df.get('data_source', '') == 'pitt_corpus']),
            "total_samples": len(df),
            "feature_groups": self.feature_groups,
            "models_trained": list(results.keys()),
            "training_results": serializable_results
        }
        
        metadata_path = self.models_dir / "enhanced_training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")


def main():
    """Main training function."""
    logger.info("Starting enhanced reminder system training...")
    
    # Initialize trainer
    trainer = EnhancedReminderSystemTrainer()
    
    # Combine datasets
    combined_df = trainer.combine_datasets()
    
    if combined_df.empty:
        logger.error("No data available for training")
        return
    
    logger.info(f"Training with {len(combined_df)} total samples")
    logger.info(f"Data sources: {combined_df['data_source'].value_counts().to_dict()}")
    
    # Train models
    results = trainer.train_enhanced_models(combined_df)
    
    # Save metadata
    trainer.save_training_metadata(results, combined_df)
    
    # Print results summary
    print("\n" + "="*50)
    print("ENHANCED TRAINING RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name}: ERROR - {result['error']}")
        else:
            print(f"{model_name}: Best Score = {result['best_score']:.4f}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()