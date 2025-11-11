"""
Data Loader Module

Handles loading sample and real dataset metadata for dementia detection system.
Seamlessly switches between sample and real data without code changes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class SampleDataLoader:
    """Loads sample dataset from JSON metadata."""

    def __init__(self, metadata_path: str = "data/sample/metadata/sample_data.json"):
        """
        Initialize sample data loader.

        Args:
            metadata_path: Path to sample_data.json
        """
        self.metadata_path = Path(metadata_path)
        self.data_root = self.metadata_path.parent.parent  # Go up to data/sample
        self.samples = []
        self.load_metadata()

    def load_metadata(self):
        """Load sample metadata from JSON file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_samples = data.get('samples', [])

                    # Resolve relative paths
                    for sample in raw_samples:
                        # Convert relative paths to absolute
                        if 'transcript_file' in sample:
                            transcript_file = sample['transcript_file']
                            full_path = self.data_root / transcript_file
                            sample['transcript_path'] = str(full_path)

                        if 'audio_file' in sample:
                            audio_file = sample['audio_file']
                            full_path = self.data_root / audio_file
                            sample['audio_path'] = str(full_path)

                    self.samples = raw_samples
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.samples = []

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all sample cases."""
        return self.samples

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific sample by ID."""
        for sample in self.samples:
            if sample.get('id') == sample_id:
                return sample
        return None

    def get_sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sample dataset."""
        control_count = sum(1 for s in self.samples if s.get('label') == 'control')
        dementia_risk_count = sum(1 for s in self.samples if s.get('label') == 'dementia_risk')

        ages = [s.get('age', 0) for s in self.samples]
        age_mean = sum(ages) / len(ages) if ages else 0
        age_range = (min(ages), max(ages)) if ages else (0, 0)

        return {
            'total_samples': len(self.samples),
            'control_count': control_count,
            'dementia_risk_count': dementia_risk_count,
            'age_mean': age_mean,
            'age_range': age_range
        }


class DatasetManager:
    """Manages dataset switching between sample and real data."""

    def __init__(self):
        """Initialize dataset manager with sample data."""
        self.sample_loader = SampleDataLoader()
        self.real_loader = None
        self.use_real_data = False

    def load_real_dataset_metadata(self, metadata_path: str):
        """
        Load real dataset metadata from path.

        Args:
            metadata_path: Path to real dataset metadata JSON
        """
        self.real_loader = SampleDataLoader(metadata_path=metadata_path)

    def switch_to_real_data(self):
        """Switch to using real dataset instead of sample."""
        if self.real_loader:
            self.use_real_data = True

    def switch_to_sample_data(self):
        """Switch to using sample dataset."""
        self.use_real_data = False

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all samples from current dataset."""
        loader = self.real_loader if self.use_real_data and self.real_loader else self.sample_loader
        return loader.get_all_samples()

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific sample by ID."""
        loader = self.real_loader if self.use_real_data and self.real_loader else self.sample_loader
        return loader.get_sample_by_id(sample_id)

    def print_dataset_info(self):
        """Print information about current dataset."""
        loader = self.real_loader if self.use_real_data and self.real_loader else self.sample_loader
        stats = loader.get_sample_statistics()

        print("\n📊 Dataset Information:")
        print(f"   Type: {'Real' if self.use_real_data else 'Sample'}")
        print(f"   Total Samples: {stats['total_samples']}")
        print(f"   Control Cases: {stats['control_count']}")
        print(f"   Dementia Risk Cases: {stats['dementia_risk_count']}")
        if stats['age_mean'] > 0:
            print(f"   Age Range: {stats['age_range'][0]}-{stats['age_range'][1]} years")
            print(f"   Mean Age: {stats['age_mean']:.1f} years")
