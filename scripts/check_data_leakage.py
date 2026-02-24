"""
Data Leakage Detection Script

Checks for common data leakage issues that cause models to show 100% accuracy:
1. Duplicate samples in train/test split
2. Target leakage (features that directly reveal the label)
3. Temporal leakage (future information in training)
4. Group leakage (same participant/session in train and test)
5. Feature correlation analysis

Usage:
    python scripts/check_data_leakage.py --data-file data/enhanced_training_data.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """
    Detects various forms of data leakage that could cause artificially high accuracy.
    """
    
    def __init__(self, data_file: str, output_dir: str = "leakage_analysis"):
        """Initialize detector."""
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.issues_found = []
        
    def load_data(self) -> pd.DataFrame:
        """Load data."""
        logger.info(f"Loading data from: {self.data_file}")
        df = pd.read_csv(self.data_file)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def check_duplicate_samples(self, df: pd.DataFrame):
        """Check for duplicate samples."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING FOR DUPLICATE SAMPLES")
        logger.info("="*80)
        
        # Check exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            logger.warning(f"⚠️  Found {exact_duplicates} EXACT DUPLICATE rows!")
            self.issues_found.append({
                'type': 'exact_duplicates',
                'count': int(exact_duplicates),
                'severity': 'HIGH',
                'fix': 'Remove duplicates using df.drop_duplicates()'
            })
        else:
            logger.info("✅ No exact duplicates found")
        
        # Check feature duplicates (ignoring ID columns)
        id_cols = ['participant_id', 'reminder_id', 'timestamp', 'file_path']
        feature_cols = [col for col in df.columns if col not in id_cols]
        
        if feature_cols:
            feature_duplicates = df[feature_cols].duplicated().sum()
            if feature_duplicates > 0:
                logger.warning(f"⚠️  Found {feature_duplicates} rows with DUPLICATE FEATURES!")
                logger.warning("   These might be different samples with identical features.")
                self.issues_found.append({
                    'type': 'feature_duplicates',
                    'count': int(feature_duplicates),
                    'severity': 'MEDIUM',
                    'fix': 'Review if these are truly different samples or data collection artifacts'
                })
            else:
                logger.info("✅ No feature duplicates found")
    
    def check_target_leakage(self, df: pd.DataFrame):
        """Check for features that directly leak the target."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING FOR TARGET LEAKAGE")
        logger.info("="*80)
        
        # Identify target columns
        target_cols = ['confusion_detected', 'dementia_label', 'caregiver_alert_needed']
        target_col = None
        
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            logger.warning("No target column found, skipping target leakage check")
            return
        
        logger.info(f"Using '{target_col}' as target variable")
        
        # Calculate correlation with all numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        high_correlation_features = []
        
        for col in numeric_cols:
            if col in df.columns and target_col in df.columns:
                correlation = abs(df[col].corr(df[target_col]))
                
                if correlation > 0.95:
                    logger.warning(f"⚠️  '{col}' has SUSPICIOUS correlation: {correlation:.4f}")
                    high_correlation_features.append((col, correlation))
                    self.issues_found.append({
                        'type': 'target_leakage',
                        'feature': col,
                        'correlation': float(correlation),
                        'severity': 'CRITICAL',
                        'fix': f"Remove feature '{col}' or verify it doesn't leak target information"
                    })
                elif correlation > 0.85:
                    logger.info(f"ℹ️  '{col}' has high correlation: {correlation:.4f} (might be legitimate)")
        
        if not high_correlation_features:
            logger.info("✅ No suspicious target leakage detected")
        
        # Plot correlation heatmap
        if len(numeric_cols) > 0:
            self._plot_correlation_heatmap(df, numeric_cols, target_col)
    
    def check_temporal_leakage(self, df: pd.DataFrame):
        """Check for temporal leakage."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING FOR TEMPORAL LEAKAGE")
        logger.info("="*80)
        
        temporal_cols = ['timestamp', 'date', 'created_at', 'updated_at']
        
        found_temporal = False
        for col in temporal_cols:
            if col in df.columns:
                found_temporal = True
                logger.warning(f"⚠️  Temporal column '{col}' found in dataset")
                logger.warning(f"   Ensure this is NOT used as a feature for training")
                self.issues_found.append({
                    'type': 'temporal_column_present',
                    'column': col,
                    'severity': 'MEDIUM',
                    'fix': f"Exclude '{col}' from training features if used for prediction"
                })
        
        if not found_temporal:
            logger.info("✅ No temporal columns found")
    
    def check_group_leakage(self, df: pd.DataFrame):
        """Check for group leakage (same participant in train/test)."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING FOR GROUP LEAKAGE")
        logger.info("="*80)
        
        group_cols = ['participant_id', 'patient_id', 'user_id', 'session_id']
        
        for col in group_cols:
            if col in df.columns:
                unique_groups = df[col].nunique()
                total_samples = len(df)
                samples_per_group = total_samples / unique_groups
                
                logger.info(f"Column '{col}':")
                logger.info(f"  Unique groups: {unique_groups}")
                logger.info(f"  Total samples: {total_samples}")
                logger.info(f"  Avg samples per group: {samples_per_group:.2f}")
                
                if samples_per_group > 1.5:
                    logger.warning(f"⚠️  Multiple samples per {col}!")
                    logger.warning(f"   Use GroupKFold or GroupShuffleSplit to prevent leakage")
                    self.issues_found.append({
                        'type': 'group_leakage_risk',
                        'column': col,
                        'samples_per_group': float(samples_per_group),
                        'severity': 'HIGH',
                        'fix': f"Use GroupKFold with '{col}' to ensure same group doesn't appear in both train/test"
                    })
                else:
                    logger.info(f"✅ One sample per {col}, no group leakage risk")
    
    def check_synthetic_vs_real_data(self, df: pd.DataFrame):
        """Check distribution of synthetic vs real data."""
        logger.info("\n" + "="*80)
        logger.info("CHECKING SYNTHETIC VS REAL DATA DISTRIBUTION")
        logger.info("="*80)
        
        if 'data_source' not in df.columns:
            logger.info("No 'data_source' column found")
            return
        
        source_dist = df['data_source'].value_counts()
        logger.info(f"\nData source distribution:\n{source_dist}")
        
        # Check if synthetic data is too perfect
        if 'synthetic' in df['data_source'].values:
            synthetic_df = df[df['data_source'] == 'synthetic']
            real_df = df[df['data_source'] != 'synthetic']
            
            target_col = 'confusion_detected' if 'confusion_detected' in df.columns else 'dementia_label'
            
            if target_col in df.columns:
                # Check if synthetic data has perfect patterns
                numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != target_col][:5]  # Sample 5 features
                
                logger.info("\nChecking if synthetic data is too perfect...")
                
                for col in numeric_cols:
                    synthetic_std = synthetic_df[col].std()
                    if len(real_df) > 0:
                        real_std = real_df[col].std()
                        
                        if synthetic_std > 0 and real_std > 0:
                            std_ratio = synthetic_std / real_std
                            if std_ratio < 0.3 or std_ratio > 3.0:
                                logger.warning(f"⚠️  Feature '{col}' has very different variance:")
                                logger.warning(f"   Synthetic std: {synthetic_std:.4f}")
                                logger.warning(f"   Real std: {real_std:.4f}")
                                logger.warning(f"   Ratio: {std_ratio:.4f}")
        
        # Calculate what percentage is synthetic
        synthetic_pct = (df['data_source'] == 'synthetic').mean() * 100
        logger.info(f"\n{synthetic_pct:.1f}% of data is synthetic")
        
        if synthetic_pct > 80:
            logger.warning("⚠️  Dataset is predominantly synthetic!")
            logger.warning("   Model performance on real-world data might be worse")
            self.issues_found.append({
                'type': 'synthetic_data_dominance',
                'synthetic_percentage': float(synthetic_pct),
                'severity': 'MEDIUM',
                'fix': 'Collect more real-world data or validate heavily on held-out real data'
            })
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str], target: str):
        """Plot correlation heatmap."""
        logger.info("\nGenerating correlation heatmap...")
        
        # Select top features by correlation with target
        correlations = []
        for feat in features:
            if feat in df.columns:
                corr = abs(df[feat].corr(df[target]))
                correlations.append((feat, corr))
        
        # Sort and select top 15
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [feat for feat, _ in correlations[:15]] + [target]
        
        # Create correlation matrix
        corr_matrix = df[top_features].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap (Top 15 Features)', fontsize=14, pad=20)
        plt.tight_layout()
        
        output_file = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation heatmap saved to: {output_file}")
    
    def generate_report(self):
        """Generate leakage detection report."""
        logger.info("\n" + "="*80)
        logger.info("DATA LEAKAGE DETECTION SUMMARY")
        logger.info("="*80)
        
        if not self.issues_found:
            logger.info("✅ No data leakage issues detected!")
            logger.info("Your dataset appears clean.")
        else:
            logger.warning(f"⚠️  Found {len(self.issues_found)} potential issues:")
            
            # Group by severity
            critical = [i for i in self.issues_found if i['severity'] == 'CRITICAL']
            high = [i for i in self.issues_found if i['severity'] == 'HIGH']
            medium = [i for i in self.issues_found if i['severity'] == 'MEDIUM']
            
            if critical:
                logger.error(f"\n🔴 CRITICAL ISSUES ({len(critical)}):")
                for issue in critical:
                    logger.error(f"  - {issue['type']}: {issue.get('fix', 'Review and fix')}")
            
            if high:
                logger.warning(f"\n🟠 HIGH PRIORITY ({len(high)}):")
                for issue in high:
                    logger.warning(f"  - {issue['type']}: {issue.get('fix', 'Review and fix')}")
            
            if medium:
                logger.info(f"\n🟡 MEDIUM PRIORITY ({len(medium)}):")
                for issue in medium:
                    logger.info(f"  - {issue['type']}: {issue.get('fix', 'Review and fix')}")
        
        # Save report
        import json
        report_file = self.output_dir / "leakage_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'total_issues': len(self.issues_found),
                'issues': self.issues_found
            }, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Data Leakage Detection")
    parser.add_argument(
        '--data-file',
        default='data/enhanced_training_data.csv',
        help='Path to data file'
    )
    parser.add_argument(
        '--output-dir',
        default='leakage_analysis',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    detector = DataLeakageDetector(args.data_file, args.output_dir)
    
    # Load data
    df = detector.load_data()
    
    # Run all checks
    detector.check_duplicate_samples(df)
    detector.check_target_leakage(df)
    detector.check_temporal_leakage(df)
    detector.check_group_leakage(df)
    detector.check_synthetic_vs_real_data(df)
    
    # Generate report
    detector.generate_report()
    
    logger.info("\n✅ Leakage detection complete!")


if __name__ == "__main__":
    main()
