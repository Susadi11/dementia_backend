#!/usr/bin/env python3
"""
Generate Model Visualizations for Dashboard

Creates confusion matrices, ROC curves, and other metrics visualizations
for BERT+XGBoost and Voice XGBoost models.

Usage:
    python scripts/generate_model_visualizations.py
"""

import sys
from pathlib import Path
import pickle
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_FILE = PROJECT_ROOT / "data" / "enhanced_training_data.csv"
VISUALIZATIONS_DIR = PROJECT_ROOT / "model_dashboard" / "visualizations"
REGISTRY_FILE = PROJECT_ROOT / "models" / "models_registry.json"

# Model paths - Now using Hugging Face models
# Models are available at:
# - https://huggingface.co/susadi/dementia_bert_xgboost_model
# - https://huggingface.co/susadi/dementia_voice_model_full
# Note: To use this script, download models from Hugging Face first:
#   from huggingface_hub import snapshot_download
#   snapshot_download(repo_id="susadi/dementia_bert_xgboost_model", local_dir="models/dementia_bert_xgboost_model")
#   snapshot_download(repo_id="susadi/dementia_voice_model_full", local_dir="models/dementia_voice_model_full")
BERT_MODEL_DIR = PROJECT_ROOT / "models" / "dementia_bert_xgboost_model"
VOICE_MODEL_DIR = PROJECT_ROOT / "models" / "dementia_voice_model_full"

# Create visualizations directory
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def load_model_and_scaler(model_path, scaler_path):
    """Load pickled model and scaler using joblib"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Dementia', 'Dementia'],
                yticklabels=['No Dementia', 'Dementia'])
    plt.title(f'{title}\nConfusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {output_path.name}")


def plot_roc_curve(y_true, y_proba, title, output_path):
    """Generate and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title}\nROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve: {output_path.name}")
    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, title, output_path):
    """Generate and save precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{title}\nPrecision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PR curve: {output_path.name}")
    return pr_auc


def plot_metrics_bar(metrics_dict, title, output_path):
    """Generate and save metrics bar chart"""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    plt.ylim([0, 1.0])
    plt.title(f'{title}\nPerformance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics bar chart: {output_path.name}")


def generate_bert_xgboost_visualizations():
    """Generate visualizations for BERT+XGBoost model"""
    print("\n" + "="*60)
    print("BERT + XGBoost Model Visualizations")
    print("="*60)

    # Load model
    model_path = BERT_MODEL_DIR / "dementia_xgboost_bert_model.pkl"
    scaler_path = BERT_MODEL_DIR / "feature_scaler.pkl"
    metadata_path = BERT_MODEL_DIR / "model_metadata.json"

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None

    print(f"Loading model from {model_path.name}...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load data
    print(f"Loading data from {DATA_FILE.name}...")
    df = pd.read_csv(DATA_FILE)

    # Filter relevant features (14 linguistic features - you may need to adjust these)
    linguistic_features = [
        'semantic_incoherence', 'repeated_questions', 'self_correction',
        'low_confidence_answers', 'hesitation_pauses', 'response_coherence',
        'word_finding_difficulty', 'circumlocution', 'tangentiality',
        'semantic_drift', 'discourse_coherence', 'cognitive_risk_score',
        'confusion_detected', 'memory_issue'
    ]

    # Prepare features
    X = df[linguistic_features].fillna(0).values
    y = df['dementia_label'].values

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split data (using same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=267/(1023+267), random_state=42, stratify=y
    )

    print(f"Test set size: {len(y_test)}")

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Confusion Matrix
    cm_path = VISUALIZATIONS_DIR / "bert_xgboost_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, "BERT + XGBoost", cm_path)

    # ROC Curve
    roc_path = VISUALIZATIONS_DIR / "bert_xgboost_roc_curve.png"
    roc_auc = plot_roc_curve(y_test, y_proba, "BERT + XGBoost", roc_path)

    # Precision-Recall Curve
    pr_path = VISUALIZATIONS_DIR / "bert_xgboost_pr_curve.png"
    pr_auc = plot_precision_recall_curve(y_test, y_proba, "BERT + XGBoost", pr_path)

    # Metrics Bar Chart
    metrics_path = VISUALIZATIONS_DIR / "bert_xgboost_metrics.png"
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    plot_metrics_bar(metrics_dict, "BERT + XGBoost", metrics_path)

    return {
        "visualizations": {
            "confusion_matrix": "visualizations/bert_xgboost_confusion_matrix.png",
            "roc_curve": "visualizations/bert_xgboost_roc_curve.png",
            "pr_curve": "visualizations/bert_xgboost_pr_curve.png",
            "metrics_chart": "visualizations/bert_xgboost_metrics.png"
        },
        "additional_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }


def generate_voice_xgboost_visualizations():
    """Generate visualizations for Voice XGBoost model"""
    print("\n" + "="*60)
    print("Voice XGBoost Model Visualizations")
    print("="*60)

    # Load model
    model_path = VOICE_MODEL_DIR / "dementia_voice_xgboost_model.pkl"
    scaler_path = VOICE_MODEL_DIR / "voice_feature_scaler.pkl"
    metadata_path = VOICE_MODEL_DIR / "voice_model_metadata.json"

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None

    print(f"Loading model from {model_path.name}...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load data
    print(f"Loading data from {DATA_FILE.name}...")
    df = pd.read_csv(DATA_FILE)

    # Voice model uses: 11 linguistic + 3 audio features
    voice_features = [
        'semantic_incoherence', 'repeated_questions', 'self_correction',
        'low_confidence_answers', 'hesitation_pauses', 'response_coherence',
        'word_finding_difficulty', 'circumlocution', 'tangentiality',
        'semantic_drift', 'discourse_coherence',
        # Audio features (P5, P6, P8 from metadata)
        'pause_frequency', 'speech_rate', 'pause_duration'
    ]

    # Prepare features
    X = df[voice_features].fillna(0).values
    y = df['dementia_label'].values

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=265/(1012+265), random_state=42, stratify=y
    )

    print(f"Test set size: {len(y_test)}")

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Confusion Matrix
    cm_path = VISUALIZATIONS_DIR / "voice_xgboost_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, "Voice XGBoost", cm_path)

    # ROC Curve
    roc_path = VISUALIZATIONS_DIR / "voice_xgboost_roc_curve.png"
    roc_auc = plot_roc_curve(y_test, y_proba, "Voice XGBoost", roc_path)

    # Precision-Recall Curve
    pr_path = VISUALIZATIONS_DIR / "voice_xgboost_pr_curve.png"
    pr_auc = plot_precision_recall_curve(y_test, y_proba, "Voice XGBoost", pr_path)

    # Metrics Bar Chart
    metrics_path = VISUALIZATIONS_DIR / "voice_xgboost_metrics.png"
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    plot_metrics_bar(metrics_dict, "Voice XGBoost", metrics_path)

    return {
        "visualizations": {
            "confusion_matrix": "visualizations/voice_xgboost_confusion_matrix.png",
            "roc_curve": "visualizations/voice_xgboost_roc_curve.png",
            "pr_curve": "visualizations/voice_xgboost_pr_curve.png",
            "metrics_chart": "visualizations/voice_xgboost_metrics.png"
        },
        "additional_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }


def update_registry(bert_viz, voice_viz):
    """Update models_registry.json with visualization paths"""
    print("\n" + "="*60)
    print("Updating models_registry.json")
    print("="*60)

    with open(REGISTRY_FILE, 'r') as f:
        registry = json.load(f)

    # Find and update BERT model
    for model in registry['models']:
        if model['id'] == 'dementia_bert_xgboost':
            if bert_viz:
                model['visualizations'] = bert_viz['visualizations']
                model['metrics']['roc_auc'] = bert_viz['additional_metrics']['roc_auc']
                model['metrics']['pr_auc'] = bert_viz['additional_metrics']['pr_auc']
                print("✓ Updated BERT + XGBoost model entry")

    # Add BERT model if doesn't exist
    bert_exists = any(m['id'] == 'dementia_bert_xgboost' for m in registry['models'])
    if not bert_exists and bert_viz:
        registry['models'].append({
            "id": "dementia_bert_xgboost",
            "name": "BERT + XGBoost Dementia Detection",
            "type": "XGBoost Classifier with BERT Embeddings",
            "algorithm": "XGBoost + BERT-base-uncased",
            "purpose": "Text-based dementia detection using linguistic features + BERT embeddings",
            "file_path": "susadi/dementia_bert_xgboost_model",
            "model_source": "huggingface",
            "huggingface_url": "https://huggingface.co/susadi/dementia_bert_xgboost_model",
            "training_date": "2026-01-04",
            "metrics": {
                "accuracy": 0.8727,
                "precision": 0.9206,
                "recall": 0.9206,
                "f1_score": 0.9206,
                "roc_auc": bert_viz['additional_metrics']['roc_auc'],
                "pr_auc": bert_viz['additional_metrics']['pr_auc']
            },
            "visualizations": bert_viz['visualizations'],
            "category": "Conversational AI"
        })
        registry['total_models'] += 1
        print("✓ Added BERT + XGBoost model entry")

    # Find and update Voice model
    for model in registry['models']:
        if model['id'] == 'dementia_voice_xgboost':
            if voice_viz:
                model['visualizations'] = voice_viz['visualizations']
                model['metrics']['roc_auc'] = voice_viz['additional_metrics']['roc_auc']
                model['metrics']['pr_auc'] = voice_viz['additional_metrics']['pr_auc']
                print("✓ Updated Voice XGBoost model entry")

    # Add Voice model if doesn't exist
    voice_exists = any(m['id'] == 'dementia_voice_xgboost' for m in registry['models'])
    if not voice_exists and voice_viz:
        registry['models'].append({
            "id": "dementia_voice_xgboost",
            "name": "Voice XGBoost Dementia Detection",
            "type": "XGBoost Classifier with Audio Features",
            "algorithm": "XGBoost + Audio Signal Processing",
            "purpose": "Multi-modal dementia detection using text + audio + BERT features",
            "file_path": "susadi/dementia_voice_model_full",
            "model_source": "huggingface",
            "huggingface_url": "https://huggingface.co/susadi/dementia_voice_model_full",
            "training_date": "2026-01-06",
            "metrics": {
                "accuracy": 0.8906,
                "precision": 0.9349,
                "recall": 0.9306,
                "f1_score": 0.9327,
                "roc_auc": voice_viz['additional_metrics']['roc_auc'],
                "pr_auc": voice_viz['additional_metrics']['pr_auc']
            },
            "visualizations": voice_viz['visualizations'],
            "category": "Conversational AI"
        })
        registry['total_models'] += 1
        print("✓ Added Voice XGBoost model entry")

    # Update timestamp
    registry['last_updated'] = datetime.now().isoformat()

    # Save registry
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"✓ Registry saved to {REGISTRY_FILE.name}")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("MODEL VISUALIZATION GENERATOR")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data File: {DATA_FILE}")
    print(f"Output Directory: {VISUALIZATIONS_DIR}")
    print("="*60)

    # Generate visualizations for both models
    bert_viz = generate_bert_xgboost_visualizations()
    voice_viz = generate_voice_xgboost_visualizations()

    # Update registry
    if bert_viz or voice_viz:
        update_registry(bert_viz, voice_viz)

    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nView your dashboard at: http://localhost:8000/dashboard/")
    print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")


if __name__ == "__main__":
    main()
