"""
Master Training Pipeline for Context-Aware Smart Reminder System

Orchestrates the complete training process:
1. Generates synthetic reminder data
2. Combines with Pitt Corpus data
3. Generates voice features
4. Trains specialized models
5. Validates and saves models

Usage:
    python scripts/master_training_pipeline.py --full-pipeline
"""

import subprocess
import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generate_synthetic_reminder_data import ReminderSyntheticDataGenerator
from scripts.generate_voice_reminder_data import VoiceReminderDataGenerator
from scripts.train_reminder_models import ReminderSystemTrainer


class MasterTrainingPipeline:
    """
    Master pipeline for training context-aware reminder system models.
    
    Coordinates all steps from data generation to model training and validation.
    """
    
    def __init__(self, output_dir: str = "data/training_pipeline"):
        """Initialize the training pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.pitt_features_file = "output/pitt_text_features.csv"
        self.synthetic_data_file = self.output_dir / "synthetic_reminder_data.csv"
        self.combined_text_file = self.output_dir / "combined_text_training_data.csv"
        self.voice_features_file = self.output_dir / "voice_reminder_features.csv"
        self.multimodal_file = self.output_dir / "multimodal_training_data.csv"
        
        # Models directory
        self.models_dir = "models/reminder_system"
        
        # Pipeline state
        self.pipeline_state = {
            "started": datetime.now().isoformat(),
            "steps_completed": [],
            "current_step": None,
            "errors": []
        }
    
    def step_1_prepare_pitt_data(self) -> bool:
        """Step 1: Prepare Pitt Corpus data if not already available."""
        logger.info("Step 1: Preparing Pitt Corpus data...")
        self.pipeline_state["current_step"] = "prepare_pitt_data"
        
        try:
            if not Path(self.pitt_features_file).exists():
                logger.info("Pitt features file not found, generating...")
                
                # Run Pitt data preparation
                cmd = [
                    sys.executable, 
                    "scripts/prepare_pitt_dataset.py",
                    "--out", self.pitt_features_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
                
                if result.returncode != 0:
                    raise Exception(f"Pitt data preparation failed: {result.stderr}")
                
                logger.info(f"Pitt data prepared: {self.pitt_features_file}")
            else:
                logger.info(f"Pitt data already available: {self.pitt_features_file}")
            
            self.pipeline_state["steps_completed"].append("prepare_pitt_data")
            return True
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            self.pipeline_state["errors"].append(f"prepare_pitt_data: {str(e)}")
            return False
    
    def step_2_generate_synthetic_data(self, num_samples: int = 2000) -> bool:
        """Step 2: Generate synthetic reminder data."""
        logger.info(f"Step 2: Generating {num_samples} synthetic reminder scenarios...")
        self.pipeline_state["current_step"] = "generate_synthetic_data"
        
        try:
            # Initialize synthetic data generator
            generator = ReminderSyntheticDataGenerator(self.pitt_features_file)
            
            # Generate synthetic dataset
            synthetic_file = generator.generate_dataset(
                num_samples=num_samples,
                output_file=str(self.synthetic_data_file)
            )
            
            logger.info(f"Synthetic data generated: {synthetic_file}")
            
            # Validate generated data
            df = pd.read_csv(synthetic_file)
            logger.info(f"Generated {len(df)} synthetic samples with {len(df.columns)} features")
            
            # Log distribution
            if 'cognitive_level' in df.columns:
                distribution = df['cognitive_level'].value_counts()
                logger.info(f"Cognitive level distribution:\n{distribution}")
            
            if 'category' in df.columns:
                category_dist = df['category'].value_counts()
                logger.info(f"Category distribution:\n{category_dist}")
            
            self.pipeline_state["steps_completed"].append("generate_synthetic_data")
            return True
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            self.pipeline_state["errors"].append(f"generate_synthetic_data: {str(e)}")
            return False
    
    def step_3_combine_text_datasets(self) -> bool:
        """Step 3: Combine synthetic and Pitt Corpus datasets."""
        logger.info("Step 3: Combining synthetic and Pitt Corpus datasets...")
        self.pipeline_state["current_step"] = "combine_text_datasets"
        
        try:
            # Initialize generator to use its combine method
            generator = ReminderSyntheticDataGenerator(self.pitt_features_file)
            
            # Combine datasets
            combined_file = generator.combine_with_pitt_dataset(
                synthetic_file=str(self.synthetic_data_file),
                pitt_file=self.pitt_features_file,
                output_file=str(self.combined_text_file)
            )
            
            # Validate combined data
            df = pd.read_csv(combined_file)
            logger.info(f"Combined dataset: {len(df)} samples with {len(df.columns)} features")
            
            # Log data source distribution
            if 'data_source' in df.columns:
                source_dist = df['data_source'].value_counts()
                logger.info(f"Data source distribution:\n{source_dist}")
            
            self.pipeline_state["steps_completed"].append("combine_text_datasets")
            return True
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            self.pipeline_state["errors"].append(f"combine_text_datasets: {str(e)}")
            return False
    
    def step_4_generate_voice_features(self, generate_audio: bool = False) -> bool:
        """Step 4: Generate voice features for multimodal training."""
        logger.info("Step 4: Generating voice features...")
        self.pipeline_state["current_step"] = "generate_voice_features"
        
        try:
            # Initialize voice generator
            voice_generator = VoiceReminderDataGenerator(
                output_dir=str(self.output_dir / "audio_samples")
            )
            
            # Generate voice features
            audio_dir = None
            if generate_audio:
                audio_dir = str(self.output_dir / "audio_files")
            
            voice_features_file = voice_generator.generate_voice_dataset(
                text_data_file=str(self.combined_text_file),
                output_file=str(self.voice_features_file),
                audio_dir=audio_dir
            )
            
            # Combine text and voice features
            multimodal_file = voice_generator.combine_text_and_voice_features(
                text_features_file=str(self.combined_text_file),
                voice_features_file=voice_features_file,
                output_file=str(self.multimodal_file)
            )
            
            # Validate multimodal data
            df = pd.read_csv(multimodal_file)
            logger.info(f"Multimodal dataset: {len(df)} samples with {len(df.columns)} features")
            
            self.pipeline_state["steps_completed"].append("generate_voice_features")
            return True
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            self.pipeline_state["errors"].append(f"generate_voice_features: {str(e)}")
            return False
    
    def step_5_train_models(self) -> bool:
        """Step 5: Train all reminder system models."""
        logger.info("Step 5: Training reminder system models...")
        self.pipeline_state["current_step"] = "train_models"
        
        try:
            # Initialize trainer
            trainer = ReminderSystemTrainer(self.models_dir)
            
            # Train models on multimodal data
            training_results = trainer.train_all_models(str(self.multimodal_file))
            
            # Log results
            logger.info("Model training results:")
            for model_name, result in training_results.items():
                if 'error' in result:
                    logger.warning(f"  {model_name}: FAILED - {result['error']}")
                else:
                    logger.info(f"  {model_name}: SUCCESS")
                    if 'auc_score' in result:
                        logger.info(f"    AUC Score: {result['auc_score']:.3f}")
                    if 'r2_score' in result:
                        logger.info(f"    R² Score: {result['r2_score']:.3f}")
                    if 'accuracy' in result:
                        logger.info(f"    Accuracy: {result['accuracy']:.3f}")
            
            # Save training results to pipeline state
            self.pipeline_state["training_results"] = training_results
            self.pipeline_state["steps_completed"].append("train_models")
            return True
            
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")
            self.pipeline_state["errors"].append(f"train_models: {str(e)}")
            return False
    
    def step_6_validate_models(self) -> bool:
        """Step 6: Validate trained models."""
        logger.info("Step 6: Validating trained models...")
        self.pipeline_state["current_step"] = "validate_models"
        
        try:
            # Load and test models
            trainer = ReminderSystemTrainer(self.models_dir)
            trainer.load_trained_models()
            
            # Test with sample data
            df = pd.read_csv(str(self.multimodal_file))
            test_sample = df.sample(min(100, len(df)))
            
            validation_results = {}
            
            # Test each model if available
            for model_name in ['confusion_detection', 'cognitive_risk', 'caregiver_alert']:
                if model_name in trainer.models:
                    try:
                        model = trainer.models[model_name]
                        
                        # Prepare test features (simplified)
                        feature_cols = [col for col in df.columns if col.startswith('feature_') or 
                                      col in ['hesitation_pauses', 'semantic_incoherence', 'response_time_seconds']]
                        
                        if feature_cols:
                            X_test = test_sample[feature_cols].fillna(0)
                            
                            if hasattr(model, 'predict'):
                                predictions = model.predict(X_test)
                                validation_results[model_name] = {
                                    'predictions_count': len(predictions),
                                    'unique_predictions': len(set(predictions)) if hasattr(predictions, '__iter__') else 1,
                                    'status': 'success'
                                }
                            else:
                                validation_results[model_name] = {'status': 'no_predict_method'}
                        else:
                            validation_results[model_name] = {'status': 'no_features'}
                            
                    except Exception as e:
                        validation_results[model_name] = {'status': 'error', 'error': str(e)}
                else:
                    validation_results[model_name] = {'status': 'model_not_found'}
            
            logger.info("Model validation results:")
            for model_name, result in validation_results.items():
                logger.info(f"  {model_name}: {result['status']}")
                if result['status'] == 'success':
                    logger.info(f"    Predictions: {result['predictions_count']}")
            
            self.pipeline_state["validation_results"] = validation_results
            self.pipeline_state["steps_completed"].append("validate_models")
            return True
            
        except Exception as e:
            logger.error(f"Step 6 failed: {e}")
            self.pipeline_state["errors"].append(f"validate_models: {str(e)}")
            return False
    
    def run_full_pipeline(
        self, 
        num_samples: int = 2000, 
        generate_audio: bool = False,
        skip_existing: bool = True
    ) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            num_samples: Number of synthetic samples to generate
            generate_audio: Whether to generate actual audio files
            skip_existing: Whether to skip steps if output files exist
            
        Returns:
            True if pipeline completed successfully
        """
        logger.info("Starting master training pipeline for context-aware reminder system")
        logger.info("="*80)
        
        pipeline_steps = [
            ("Prepare Pitt Corpus Data", lambda: self.step_1_prepare_pitt_data()),
            ("Generate Synthetic Data", lambda: self.step_2_generate_synthetic_data(num_samples)),
            ("Combine Text Datasets", lambda: self.step_3_combine_text_datasets()),
            ("Generate Voice Features", lambda: self.step_4_generate_voice_features(generate_audio)),
            ("Train Models", lambda: self.step_5_train_models()),
            ("Validate Models", lambda: self.step_6_validate_models())
        ]
        
        success = True
        
        for i, (step_name, step_func) in enumerate(pipeline_steps, 1):
            logger.info(f"\n{'='*20} Step {i}/6: {step_name} {'='*20}")
            
            try:
                if not step_func():
                    success = False
                    logger.error(f"Step {i} failed, stopping pipeline")
                    break
                else:
                    logger.info(f"Step {i} completed successfully ✅")
                    
            except Exception as e:
                logger.error(f"Step {i} crashed: {e}")
                success = False
                break
        
        # Save pipeline state
        self.pipeline_state["completed"] = datetime.now().isoformat()
        self.pipeline_state["success"] = success
        
        pipeline_log = self.output_dir / "pipeline_log.json"
        with open(pipeline_log, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*80)
        if success:
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"✅ All {len(pipeline_steps)} steps completed")
            logger.info(f"📊 Training data: {self.multimodal_file}")
            logger.info(f"🤖 Models saved to: {self.models_dir}")
            logger.info(f"📝 Pipeline log: {pipeline_log}")
        else:
            logger.error("❌ PIPELINE FAILED")
            logger.error(f"Completed steps: {self.pipeline_state['steps_completed']}")
            logger.error(f"Errors: {self.pipeline_state['errors']}")
        
        logger.info("="*80)
        
        return success
    
    def print_usage_instructions(self):
        """Print instructions for using the trained models."""
        if not self.pipeline_state.get("success", False):
            logger.warning("Pipeline did not complete successfully")
            return
        
        print("\n" + "="*60)
        print("🚀 HOW TO USE YOUR TRAINED REMINDER SYSTEM")
        print("="*60)
        
        print("\n1. 📝 START THE API SERVER:")
        print("   python src/api/app_simple.py")
        
        print("\n2. 🌐 TEST THE REAL-TIME SYSTEM:")
        print("   python test_realtime_system.py")
        print("   # OR open test_realtime_web.html in browser")
        
        print("\n3. 🧠 USE THE MODELS PROGRAMMATICALLY:")
        print(f"""
   from scripts.train_reminder_models import ReminderSystemTrainer
   
   # Load trained models
   trainer = ReminderSystemTrainer("{self.models_dir}")
   trainer.load_trained_models()
   
   # Use for prediction
   # trainer.models['confusion_detection'].predict(features)
   # trainer.models['cognitive_risk'].predict(features)
   # trainer.models['caregiver_alert'].predict(features)
        """)
        
        print("\n4. 📊 GENERATED DATASETS:")
        print(f"   - Synthetic data: {self.synthetic_data_file}")
        print(f"   - Combined text: {self.combined_text_file}")  
        print(f"   - Voice features: {self.voice_features_file}")
        print(f"   - Multimodal: {self.multimodal_file}")
        
        print("\n5. 🤖 TRAINED MODELS:")
        print(f"   - Models directory: {self.models_dir}")
        print("   - confusion_detection_model.joblib")
        print("   - cognitive_risk_model.joblib")
        print("   - caregiver_alert_model.joblib")
        print("   - response_classifier_model.joblib")
        
        print("\n" + "="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Master training pipeline for context-aware smart reminder system'
    )
    
    parser.add_argument(
        '--full-pipeline', 
        action='store_true', 
        help='Run the complete training pipeline'
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=2000, 
        help='Number of synthetic samples to generate (default: 2000)'
    )
    parser.add_argument(
        '--generate-audio', 
        action='store_true', 
        help='Generate actual audio files (slower but more realistic)'
    )
    parser.add_argument(
        '--output-dir', 
        default='data/training_pipeline', 
        help='Output directory for pipeline files'
    )
    
    args = parser.parse_args()
    
    if not args.full_pipeline:
        print("Use --full-pipeline to run the complete training process")
        parser.print_help()
        return
    
    # Initialize and run pipeline
    pipeline = MasterTrainingPipeline(args.output_dir)
    
    success = pipeline.run_full_pipeline(
        num_samples=args.num_samples,
        generate_audio=args.generate_audio
    )
    
    if success:
        pipeline.print_usage_instructions()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()