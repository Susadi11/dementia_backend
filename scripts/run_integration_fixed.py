"""
Fixed Integration Runner for Pitt Corpus

This is a more robust version that handles common errors and provides
better debugging information.
"""

import subprocess
import sys
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_pitt_data():
    """Check if Pitt data is available and accessible."""
    pitt_dir = Path("data/Pitt")
    
    if not pitt_dir.exists():
        logger.error(f"❌ Pitt Corpus data not found at {pitt_dir}")
        return False
    
    # Check for Control and Dementia directories
    control_dir = pitt_dir / "Control"
    dementia_dir = pitt_dir / "Dementia"
    
    if not control_dir.exists():
        logger.error("❌ Control directory not found")
        return False
        
    if not dementia_dir.exists():
        logger.error("❌ Dementia directory not found")
        return False
    
    # Count .cha files
    control_files = len(list(control_dir.rglob("*.cha")))
    dementia_files = len(list(dementia_dir.rglob("*.cha")))
    
    logger.info(f"✅ Found {control_files} Control .cha files")
    logger.info(f"✅ Found {dementia_files} Dementia .cha files")
    
    if control_files == 0 or dementia_files == 0:
        logger.error("❌ No .cha files found in Pitt directories")
        return False
    
    return True


def run_command_safe(cmd: str, description: str):
    """Run a command with better error handling."""
    logger.info(f"🔄 {description}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent.parent,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed")
            return True
        else:
            logger.error(f"❌ {description} failed (exit code: {result.returncode})")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False


def create_minimal_integration():
    """Create a minimal integration that just combines existing data."""
    logger.info("📋 Creating minimal integration using synthetic data only...")
    
    try:
        # Load synthetic data
        synthetic_df = pd.read_csv("data/synthetic_reminder_data.csv")
        logger.info(f"✅ Loaded {len(synthetic_df)} synthetic samples")
        
        # Add data source column
        synthetic_df['data_source'] = 'synthetic'
        
        # Save as enhanced training data
        output_file = "data/enhanced_training_data.csv"
        synthetic_df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved enhanced dataset to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create minimal integration: {e}")
        return False


def test_integration():
    """Test if the integration was successful."""
    try:
        # Check if enhanced data file exists
        enhanced_file = Path("data/enhanced_training_data.csv")
        if not enhanced_file.exists():
            return False
        
        # Check if it has data
        df = pd.read_csv(enhanced_file)
        if len(df) == 0:
            return False
        
        logger.info(f"✅ Enhanced dataset has {len(df)} samples")
        
        if 'data_source' in df.columns:
            logger.info(f"Data sources: {df['data_source'].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main function with improved error handling."""
    logger.info("🚀 Starting Fixed Pitt Corpus Integration...")
    
    # Step 1: Check if Pitt data is available
    if not check_pitt_data():
        logger.warning("⚠️  Pitt Corpus not available, using synthetic data only")
        if create_minimal_integration():
            logger.info("✅ Minimal integration completed")
        else:
            logger.error("❌ Even minimal integration failed")
            return
    else:
        # Step 2: Try to extract Pitt features with better error handling
        logger.info("📊 Extracting features from Pitt Corpus...")
        
        # Use a simpler extraction approach
        extract_cmd = 'python -c "import scripts.integrate_pitt_data as ipd; integrator = ipd.PittDataIntegrator(); integrator.extract_pitt_features(\\"data/pitt_features.csv\\")"'
        
        if not run_command_safe(extract_cmd, "Extract Pitt features"):
            logger.warning("⚠️  Pitt extraction failed, falling back to synthetic only")
            if not create_minimal_integration():
                logger.error("❌ Fallback integration failed")
                return
        else:
            # Step 3: Create balanced dataset
            if not run_command_safe(
                "python scripts/integrate_pitt_data.py --create-balanced --output data/enhanced_training_data.csv",
                "Create balanced dataset"
            ):
                logger.warning("⚠️  Balanced dataset creation failed, using synthetic only")
                if not create_minimal_integration():
                    logger.error("❌ Fallback integration failed")
                    return
    
    # Step 4: Test the integration
    if not test_integration():
        logger.error("❌ Integration validation failed")
        return
    
    # Step 5: Train models
    logger.info("🎯 Training enhanced models...")
    if run_command_safe(
        "python scripts/train_enhanced_reminder_models.py",
        "Train enhanced models"
    ):
        logger.info("🎉 Integration and training completed successfully!")
        
        # Show summary
        print("\n" + "="*60)
        print("INTEGRATION COMPLETED")
        print("="*60)
        print("✅ Dataset prepared")
        print("✅ Models trained")
        print("\nNext steps:")
        print("- Run: python scripts/test_enhanced_models.py")
        print("- Check models in: models/reminder_system/")
        print("- Review data in: data/enhanced_training_data.csv")
    else:
        logger.error("❌ Model training failed")
        
        # Still provide guidance for manual steps
        print("\n" + "="*60)
        print("PARTIAL SUCCESS - DATA READY")
        print("="*60)
        print("✅ Dataset created: data/enhanced_training_data.csv")
        print("❌ Model training failed")
        print("\nManual next steps:")
        print("1. Check data quality: python scripts/integrate_pitt_data.py --validate --output data/enhanced_training_data.csv")
        print("2. Try training again: python scripts/train_enhanced_reminder_models.py")


if __name__ == "__main__":
    main()