"""
Comprehensive Model Validation for Context-Aware Smart Reminder System

This script helps identify overfitting and provides proper validation metrics:
1. K-Fold Cross-Validation (prevents overfitting detection)
2. Learning Curves (shows if more data needed)
3. Confusion Matrix Analysis
4. Feature Importance Analysis
5. Data Distribution Analysis
6. Stratified Train/Validation/Test Split

Usage:
    python scripts/validate_models_comprehensive.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, learning_curve, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveModelValidator:
    """
    Comprehensive validation suite for dementia detection models.
    
    Detects overfitting, data leakage, and provides actionable recommendations.
    """
    
    def __init__(self, data_file: str, output_dir: str = "validation_results"):
        """Initialize validator."""
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'data_file': data_file,
            'validation_results': {},
            'recommendations': [],
            'overfitting_detected': False
        }
        
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load and analyze data distribution."""
        logger.info(f"Loading data from {self.data_file}")
        df = pd.read_csv(self.data_file)
        
        logger.info(f"Total samples: {len(df)}")
        
        # Check data sources
        if 'data_source' in df.columns:
            source_counts = df['data_source'].value_counts()
            logger.info(f"\nData sources:\n{source_counts}")
            self.report['data_sources'] = source_counts.to_dict()
        
        # Check label distribution
        label_col = 'confusion_detected' if 'confusion_detected' in df.columns else 'dementia_label'
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            logger.info(f"\nLabel distribution ({label_col}):\n{label_counts}")
            self.report['label_distribution'] = label_counts.to_dict()
            
            # Check for class imbalance
            class_ratio = label_counts.min() / label_counts.max()
            if class_ratio < 0.3:
                logger.warning(f"⚠️  CLASS IMBALANCE DETECTED! Ratio: {class_ratio:.2f}")
                self.report['recommendations'].append(
                    f"Class imbalance detected (ratio: {class_ratio:.2f}). Consider using SMOTE or class weights."
                )
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠️  {duplicates} duplicate rows found!")
            self.report['recommendations'].append(
                f"Remove {duplicates} duplicate rows to prevent data leakage."
            )
        
        # Check for data leakage risks
        self._check_data_leakage_risks(df)
        
        return df
    
    def _check_data_leakage_risks(self, df: pd.DataFrame):
        """Check for potential data leakage issues."""
        logger.info("\n🔍 Checking for data leakage risks...")
        
        leakage_risks = []
        
        # Check if same participant appears in multiple rows
        if 'participant_id' in df.columns:
            participant_counts = df['participant_id'].value_counts()
            repeated_participants = (participant_counts > 1).sum()
            if repeated_participants > 0:
                leakage_risks.append(
                    f"Same participant appears in multiple samples ({repeated_participants} participants). "
                    "Use GroupKFold instead of regular cross-validation to prevent leakage."
                )
        
        # Check if timestamp-based features could leak
        if 'timestamp' in df.columns:
            leakage_risks.append(
                "Timestamp column present - ensure it's not used as a feature."
            )
        
        # Check for suspiciously perfect correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        label_col = 'confusion_detected' if 'confusion_detected' in df.columns else 'dementia_label'
        
        if label_col in numeric_cols:
            for col in numeric_cols:
                if col != label_col and col in df.columns:
                    correlation = abs(df[col].corr(df[label_col]))
                    if correlation > 0.95:
                        leakage_risks.append(
                            f"Feature '{col}' has suspiciously high correlation ({correlation:.3f}) with label. "
                            "This might indicate data leakage."
                        )
        
        if leakage_risks:
            logger.warning("⚠️  POTENTIAL DATA LEAKAGE DETECTED:")
            for risk in leakage_risks:
                logger.warning(f"  - {risk}")
            self.report['data_leakage_risks'] = leakage_risks
        else:
            logger.info("✅ No obvious data leakage detected")
    
    def validate_confusion_detection_model(self, df: pd.DataFrame):
        """Validate confusion detection model with proper cross-validation."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATING CONFUSION DETECTION MODEL")
        logger.info("="*80)
        
        # Prepare features
        text_features = [
            "hesitation_pauses", "semantic_incoherence", "low_confidence_answers",
            "repeated_questions", "self_correction", "response_coherence"
        ]
        temporal_features = ["response_time_seconds", "pause_frequency", "speech_rate"]
        
        available_features = [f for f in text_features + temporal_features if f in df.columns]
        
        if not available_features:
            logger.error("No required features found!")
            return
        
        X = df[available_features].fillna(0).values
        y = df['confusion_detected'].astype(int).values if 'confusion_detected' in df.columns else df['dementia_label'].astype(int).values
        
        logger.info(f"Features used: {available_features}")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Test multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {model_name}")
            logger.info(f"{'='*60}")
            
            # Stratified K-Fold Cross-Validation
            cv_results = self._cross_validate_model(model, X, y, model_name)
            
            # Learning Curve Analysis
            learning_results = self._plot_learning_curve(model, X, y, model_name)
            
            # Train final model for feature importance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if 'Logistic' in model_name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_train = model.predict(X_train_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred)
            
            logger.info(f"\nTrain Accuracy: {train_acc:.4f}")
            logger.info(f"Test Accuracy:  {test_acc:.4f}")
            
            # Check for overfitting
            overfitting_gap = train_acc - test_acc
            if overfitting_gap > 0.15:
                logger.warning(f"⚠️  OVERFITTING DETECTED! Gap: {overfitting_gap:.4f}")
                self.report['overfitting_detected'] = True
                self.report['recommendations'].append(
                    f"{model_name}: Overfitting gap of {overfitting_gap:.4f}. "
                    "Consider: 1) Increase regularization, 2) Reduce model complexity, "
                    "3) Add more diverse data, 4) Use dropout or early stopping."
                )
            elif test_acc > 0.95:
                logger.warning(f"⚠️  Suspiciously high test accuracy ({test_acc:.4f}). Check for data leakage!")
                self.report['recommendations'].append(
                    f"{model_name}: Test accuracy {test_acc:.4f} is suspiciously high. "
                    "Verify there's no data leakage."
                )
            else:
                logger.info(f"✅ No significant overfitting detected")
            
            # Detailed metrics
            logger.info(f"\nDetailed Test Set Metrics:")
            logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
            logger.info(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
            logger.info(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  {cm}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(available_features, model.feature_importances_))
                sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"\nTop 5 Important Features:")
                for feat, imp in sorted_importances[:5]:
                    logger.info(f"  {feat}: {imp:.4f}")
            
            results[model_name] = {
                'cv_results': cv_results,
                'learning_results': learning_results,
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'overfitting_gap': float(overfitting_gap),
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        
        self.report['validation_results']['confusion_detection'] = results
        
        # Plot comparison
        self._plot_model_comparison(results, "Confusion Detection Models")
        
        return results
    
    def _cross_validate_model(self, model, X, y, model_name: str, n_folds: int = 5) -> Dict:
        """Perform stratified k-fold cross-validation."""
        logger.info(f"\nPerforming {n_folds}-Fold Cross-Validation...")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
        )
        
        # Calculate mean and std for each metric
        results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[metric] = {
                'test_mean': float(np.mean(test_scores)),
                'test_std': float(np.std(test_scores)),
                'train_mean': float(np.mean(train_scores)),
                'train_std': float(np.std(train_scores)),
                'gap': float(np.mean(train_scores) - np.mean(test_scores))
            }
            
            logger.info(f"  {metric.capitalize()}:")
            logger.info(f"    Train: {results[metric]['train_mean']:.4f} (±{results[metric]['train_std']:.4f})")
            logger.info(f"    Test:  {results[metric]['test_mean']:.4f} (±{results[metric]['test_std']:.4f})")
            logger.info(f"    Gap:   {results[metric]['gap']:.4f}")
        
        return results
    
    def _plot_learning_curve(self, model, X, y, model_name: str, cv: int = 5):
        """Generate learning curve to diagnose bias/variance."""
        logger.info(f"\nGenerating learning curve...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
            scoring='accuracy', random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes_abs, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add interpretation
        final_gap = train_mean[-1] - test_mean[-1]
        if final_gap > 0.15:
            plt.text(0.5, 0.05, 'HIGH VARIANCE (Overfitting)\nRecommendation: More data or regularization',
                    transform=plt.gca().transAxes, fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif test_mean[-1] < 0.7:
            plt.text(0.5, 0.05, 'HIGH BIAS (Underfitting)\nRecommendation: More complex model or features',
                    transform=plt.gca().transAxes, fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        output_file = self.output_dir / f"learning_curve_{model_name.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Learning curve saved to: {output_file}")
        
        return {
            'train_mean': train_mean.tolist(),
            'test_mean': test_mean.tolist(),
            'final_gap': float(final_gap)
        }
    
    def _plot_model_comparison(self, results: Dict, title: str):
        """Plot comparison of different models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{title} - Performance Comparison', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            train_scores = [results[m]['cv_results'][metric]['train_mean'] for m in model_names]
            test_scores = [results[m]['cv_results'][metric]['test_mean'] for m in model_names]
            test_stds = [results[m]['cv_results'][metric]['test_std'] for m in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8, yerr=test_stds)
            
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Scores')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        output_file = self.output_dir / f"model_comparison_{title.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nModel comparison plot saved to: {output_file}")
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("="*80)
        
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        logger.info(f"\nFull report saved to: {report_file}")
        
        # Print recommendations
        if self.report['recommendations']:
            logger.info("\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(self.report['recommendations'], 1):
                logger.info(f"{i}. {rec}")
        
        # Print final verdict
        if self.report['overfitting_detected']:
            logger.warning("\n⚠️  VERDICT: Models showing signs of overfitting!")
            logger.warning("Action required: Fine-tune models with recommended changes.")
        else:
            logger.info("\n✅ VERDICT: Models appear to be generalizing well.")
        
        return self.report


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Validation")
    parser.add_argument(
        '--data-file',
        default='data/enhanced_training_data.csv',
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--output-dir',
        default='validation_results',
        help='Directory to save validation results'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = ComprehensiveModelValidator(args.data_file, args.output_dir)
    
    # Load and analyze data
    df = validator.load_and_analyze_data()
    
    # Validate confusion detection model
    validator.validate_confusion_detection_model(df)
    
    # Generate final report
    validator.generate_report()
    
    logger.info("\n✅ Validation complete!")
    logger.info(f"Check results in: {validator.output_dir}")


if __name__ == "__main__":
    main()
