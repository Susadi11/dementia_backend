"""
Automated Model Registry Scanner
Scans the backend for all trained models and extracts their metrics
Generates a models_registry.json file for the dashboard
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import sys

# Try to import joblib, but continue if not available
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("⚠️  Warning: joblib not available, some models may not be loaded")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelRegistryScanner:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.models_registry = {
            "last_updated": datetime.now().isoformat(),
            "total_models": 0,
            "models": []
        }

    def get_file_size_mb(self, file_path):
        """Get file size in MB"""
        size_bytes = os.path.getsize(file_path)
        return round(size_bytes / (1024 * 1024), 2)

    def extract_model_type(self, model_obj):
        """Extract model type from the model object"""
        model_type = type(model_obj).__name__
        if hasattr(model_obj, '__class__'):
            return model_obj.__class__.__name__
        return model_type

    def scan_reminder_system_models(self):
        """Scan reminder system models and extract metrics"""
        reminder_path = self.base_path / "models" / "reminder_system"

        if not reminder_path.exists():
            return

        # Load training metadata
        metadata_file = reminder_path / "training_metadata.json"
        enhanced_metadata_file = reminder_path / "enhanced_training_metadata.json"

        metadata = {}
        if enhanced_metadata_file.exists():
            with open(enhanced_metadata_file, 'r') as f:
                metadata = json.load(f)
        elif metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Model configurations
        model_configs = [
            {
                "name": "Confusion Detection Model",
                "file": "confusion_detection_model.joblib",
                "scaler": "confusion_detection_scaler.joblib",
                "type": "Binary Classifier",
                "purpose": "Detects confusion in patient responses",
                "metrics_key": "confusion_detection"
            },
            {
                "name": "Cognitive Risk Assessment Model",
                "file": "cognitive_risk_model.joblib",
                "scaler": "cognitive_risk_scaler.joblib",
                "type": "Regressor",
                "purpose": "Predicts cognitive risk score (0-1)",
                "metrics_key": "cognitive_risk"
            },
            {
                "name": "Caregiver Alert Model",
                "file": "caregiver_alert_model.joblib",
                "scaler": "caregiver_alert_scaler.joblib",
                "type": "Binary Classifier",
                "purpose": "Determines if caregiver needs to be alerted",
                "metrics_key": "caregiver_alert"
            },
            {
                "name": "Response Classifier Model",
                "file": "response_classifier_model.joblib",
                "scaler": "response_classifier_scaler.joblib",
                "type": "Multi-class Classifier",
                "purpose": "Classifies patient response type (5 classes)",
                "metrics_key": "response_classifier"
            }
        ]

        for config in model_configs:
            model_file = reminder_path / config["file"]
            if not model_file.exists():
                continue

            try:
                # Load model to get type
                if HAS_JOBLIB:
                    model = joblib.load(model_file)
                    model_type_detail = self.extract_model_type(model)
                else:
                    model_type_detail = "Unknown (joblib not available)"

                # Extract metrics from metadata
                metrics = {}
                dataset_info = {}

                # Try to get training results and dataset info
                if metadata:
                    if "training_results" in metadata:
                        training_results = metadata["training_results"].get(config["metrics_key"], {})
                        if training_results:
                            metrics = {
                                "best_score": training_results.get("best_score", "Unknown"),
                                "algorithm_results": training_results.get("results", {}),
                                "feature_count": training_results.get("feature_count", "Unknown")
                            }

                    # Get dataset info
                    dataset_info = {
                        "total_samples": metadata.get("total_samples", "Unknown"),
                        "synthetic_samples": metadata.get("synthetic_samples", 0),
                        "pitt_samples": metadata.get("pitt_samples", 0),
                        "data_sources": metadata.get("data_sources", [])
                    }

                # Build model entry
                model_entry = {
                    "id": config["metrics_key"],
                    "name": config["name"],
                    "type": config["type"],
                    "algorithm": model_type_detail,
                    "purpose": config["purpose"],
                    "file_path": str(model_file.relative_to(self.base_path)),
                    "file_size_mb": self.get_file_size_mb(model_file),
                    "has_scaler": (reminder_path / config["scaler"]).exists(),
                    "training_date": metadata.get("training_date", "Unknown"),
                    "dataset_info": dataset_info,
                    "metrics": metrics,
                    "category": "Reminder System"
                }

                self.models_registry["models"].append(model_entry)

            except Exception as e:
                print(f"Error processing {config['name']}: {str(e)}")

    def scan_text_models(self):
        """Scan text-based conversational AI models"""
        models_path = self.base_path / "models"

        text_model_files = ["text_model.pkl", "text_model_sample.pkl"]

        for model_file_name in text_model_files:
            model_file = models_path / model_file_name
            if not model_file.exists():
                continue

            try:
                # Load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

                model_type_detail = self.extract_model_type(model)

                model_entry = {
                    "id": model_file_name.replace(".pkl", ""),
                    "name": f"Text-based Dementia Detection Model ({model_file_name})",
                    "type": "Binary Classifier",
                    "algorithm": model_type_detail,
                    "purpose": "Detects dementia from text-based conversation features",
                    "file_path": str(model_file.relative_to(self.base_path)),
                    "file_size_mb": self.get_file_size_mb(model_file),
                    "has_scaler": False,
                    "training_date": "Unknown",
                    "dataset_size": "Unknown",
                    "num_features": 7,  # TEXT_FEATURES from text_model_trainer.py
                    "features_used": [
                        "semantic_incoherence",
                        "repeated_questions",
                        "self_correction",
                        "low_confidence_answers",
                        "hesitation_pauses",
                        "emotion_slip",
                        "evening_errors"
                    ],
                    "metrics": {},
                    "category": "Conversational AI"
                }

                self.models_registry["models"].append(model_entry)

            except Exception as e:
                print(f"Error processing {model_file_name}: {str(e)}")

    def scan_game_models(self):
        """Scan game risk classifier models"""
        game_path = self.base_path / "src" / "models" / "game" / "risk_classifier"

        if not game_path.exists():
            return

        logistic_model = game_path / "logistic_model.pkl"
        scaler_file = game_path / "scaler.pkl"

        if logistic_model.exists():
            try:
                with open(logistic_model, 'rb') as f:
                    model = pickle.load(f)

                model_type_detail = self.extract_model_type(model)

                model_entry = {
                    "id": "game_risk_classifier",
                    "name": "Game Risk Classifier Model",
                    "type": "Binary Classifier",
                    "algorithm": model_type_detail,
                    "purpose": "Classifies cognitive risk from game performance data",
                    "file_path": str(logistic_model.relative_to(self.base_path)),
                    "file_size_mb": self.get_file_size_mb(logistic_model),
                    "has_scaler": scaler_file.exists(),
                    "training_date": "Unknown",
                    "dataset_size": "Unknown",
                    "num_features": "Unknown",
                    "metrics": {},
                    "category": "Game Analysis"
                }

                self.models_registry["models"].append(model_entry)

            except Exception as e:
                print(f"Error processing game model: {str(e)}")

    def add_llama_model(self):
        """Add LLaMA 3.2 3B fine-tuned model info"""
        llama_entry = {
            "id": "llama_3_2_3b_dementia_care",
            "name": "LLaMA 3.2 3B Dementia Care Chatbot",
            "type": "Large Language Model (Fine-tuned)",
            "algorithm": "LLaMA 3.2 3B with LoRA Adapter",
            "purpose": "Conversational chatbot for empathetic dementia patient care",
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "lora_adapter": "susadi/llama-3.2-3b-dementia-care",
            "huggingface_url": "https://huggingface.co/susadi/llama-3.2-3b-dementia-care",
            "file_path": "Cloud-hosted (HuggingFace)",
            "file_size_mb": "~3000 (3B parameters)",
            "training_dataset": "DailyDialog",
            "training_method": "LoRA (Low-Rank Adaptation)",
            "features": [
                "Multi-turn conversation support",
                "Context-aware responses",
                "GPU/MPS/CPU inference",
                "Integrated with NLPEngine for intelligent detection"
            ],
            "inference_hardware": ["CUDA (GPU)", "MPS (Apple Silicon)", "CPU"],
            "training_date": "2024-12",
            "metrics": {
                "model_info": {
                    "parameters": "3 billion",
                    "quantization": "4-bit (optional for efficiency)",
                    "fine_tuning_method": "LoRA (Low-Rank Adaptation)"
                },
                "training_info": {
                    "base_dataset": "DailyDialog",
                    "domain_adaptation": "Dementia care conversation",
                    "training_samples": "Conversational dialogues",
                    "note": "Fine-tuned for empathetic, context-aware responses"
                },
                "evaluation_metrics": {
                    "response_quality": "Optimized for empathy and clarity",
                    "context_awareness": "Multi-turn conversation support",
                    "deployment_ready": "Production-ready with GPU/MPS/CPU support",
                    "inference_optimization": "Supports 4-bit quantization for efficiency"
                }
            },
            "category": "Conversational AI"
        }

        self.models_registry["models"].append(llama_entry)

    def scan_all_models(self):
        """Scan all models in the backend"""
        print("🔍 Scanning for models...")

        # Scan different model categories
        self.scan_reminder_system_models()
        self.scan_text_models()
        self.scan_game_models()
        self.add_llama_model()

        # Update total count
        self.models_registry["total_models"] = len(self.models_registry["models"])

        print(f"✅ Found {self.models_registry['total_models']} models!")

    def save_registry(self, output_path):
        """Save registry to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.models_registry, f, indent=2)
        print(f"💾 Registry saved to: {output_path}")

        # Also copy to model_dashboard folder for web viewing
        dashboard_path = output_path.parent / "models" / "models_registry.json"
        if dashboard_path.parent.exists():
            import shutil
            shutil.copy(output_path, dashboard_path)
            print(f"📋 Copied to dashboard: {dashboard_path}")


def main():
    # Get base path (backend root)
    base_path = Path(__file__).parent.parent

    # Initialize scanner
    scanner = ModelRegistryScanner(base_path)

    # Scan all models
    scanner.scan_all_models()

    # Save to models_registry.json in root
    output_path = base_path / "models_registry.json"
    scanner.save_registry(output_path)

    print("\n" + "="*60)
    print("📊 Model Registry Summary:")
    print("="*60)
    print(f"Total Models: {scanner.models_registry['total_models']}")
    print(f"Last Updated: {scanner.models_registry['last_updated']}")
    print("\nModel Categories:")

    categories = {}
    for model in scanner.models_registry["models"]:
        cat = model["category"]
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in categories.items():
        print(f"  - {cat}: {count} models")

    print("="*60)


if __name__ == "__main__":
    main()
