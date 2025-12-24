"""
Quick Integration Runner

Simple script to integrate Pitt Corpus data with your reminder system.
Run this to enhance your models with real-world dementia speech data.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            logger.error(f"❌ {description} failed")
            if result.stderr:
                print("Error output:", result.stderr)
            if result.stdout:
                print("Standard output:", result.stdout)
            return False
    
    except Exception as e:
        logger.error(f"❌ Failed to run {description}: {e}")
        return False
    
    return True


def main():
    """Run the integration process."""
    logger.info("🚀 Starting Pitt Corpus integration with reminder system...")
    
    # Check if Pitt data exists
    pitt_dir = Path("data/Pitt")
    if not pitt_dir.exists():
        logger.error(f"❌ Pitt Corpus data not found at {pitt_dir}")
        logger.info("Please ensure you have downloaded the Pitt Corpus dataset")
        return
    
    # Step 1: Extract Pitt features
    logger.info("📊 Step 1: Extracting features from Pitt Corpus...")
    if not run_command(
        "python scripts/integrate_pitt_data.py --extract-pitt --pitt-features data/pitt_features.csv",
        "Extract Pitt Corpus features"
    ):
        return
    
    # Step 2: Create balanced dataset
    logger.info("⚖️ Step 2: Creating balanced training dataset...")
    if not run_command(
        "python scripts/integrate_pitt_data.py --create-balanced --output data/enhanced_training_data.csv",
        "Create balanced dataset"
    ):
        return
    
    # Step 3: Validate integration
    logger.info("🔍 Step 3: Validating integrated dataset...")
    if not run_command(
        "python scripts/integrate_pitt_data.py --validate --output data/enhanced_training_data.csv",
        "Validate integration"
    ):
        return
    
    # Step 4: Train enhanced models
    logger.info("🎯 Step 4: Training enhanced models with integrated data...")
    if not run_command(
        "python scripts/train_enhanced_reminder_models.py",
        "Train enhanced models"
    ):
        return
    
    logger.info("🎉 Integration completed successfully!")
    logger.info("Your reminder system models now include real-world dementia patterns from Pitt Corpus")
    
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    print("✅ Pitt Corpus features extracted")
    print("✅ Balanced dataset created")
    print("✅ Data integration validated")
    print("✅ Enhanced models trained")
    print("\nFiles created:")
    print("- data/pitt_features.csv (Pitt Corpus features)")
    print("- data/enhanced_training_data.csv (Combined dataset)")
    print("- data/enhanced_training_data_validation.json (Validation report)")
    print("- models/reminder_system/ (Enhanced models)")
    print("\nNext steps:")
    print("- Test your enhanced models with test_enhanced_models.py")
    print("- Deploy the new models in your reminder system")
    print("- Monitor performance improvements")


if __name__ == "__main__":
    main()