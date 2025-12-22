"""
Data Integration Utility for Pitt Corpus + Synthetic Reminder Data

This utility helps prepare, validate, and combine the Pitt Corpus data
with synthetic reminder data for training enhanced models.

Features:
- Extract and process Pitt Corpus .cha files
- Generate Pitt-derived features compatible with reminder system
- Validate data quality and feature alignment
- Create balanced datasets for training

Usage:
  python scripts/integrate_pitt_data.py --output data/integrated_training_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PittDataIntegrator:
    """Utility class for integrating Pitt Corpus data with reminder system."""
    
    def __init__(self):
        self.pitt_dir = Path("data/Pitt")
        self.synthetic_file = Path("data/synthetic_reminder_data.csv")
        
    def extract_pitt_features(self, output_file: str = "data/pitt_features.csv"):
        """Extract features from Pitt Corpus and save to CSV."""
        logger.info("Extracting features from Pitt Corpus...")
        
        if not self.pitt_dir.exists():
            logger.error(f"Pitt directory not found: {self.pitt_dir}")
            return None
        
        pitt_data = []
        
        # Process both Control and Dementia groups
        for group in ['Control', 'Dementia']:
            group_dir = self.pitt_dir / group
            if not group_dir.exists():
                continue
            
            dementia_label = 1 if group == 'Dementia' else 0
            logger.info(f"Processing {group} group...")
            
            # Process each task type
            for task_dir in group_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                task_name = task_dir.name
                cha_files = list(task_dir.glob("*.cha"))
                logger.info(f"  {task_name}: {len(cha_files)} files")
                
                for cha_file in cha_files:
                    try:
                        # Extract participant text
                        text = self._extract_participant_text(cha_file)
                        if not text.strip():
                            continue
                        
                        # Extract features with error handling
                        try:
                            features = self._extract_comprehensive_features(text, task_name)
                        except Exception as e:
                            logger.warning(f"Feature extraction failed for {cha_file.name}: {e}")
                            # Use basic fallback features
                            features = self._get_zero_features()
                            features.update({
                                'word_count': len(text.split()) if text else 0,
                                'sentence_count': len(text.split('.')) if text else 0
                            })
                        
                        # Add metadata
                        features.update({
                            'participant_id': cha_file.stem,
                            'task_type': task_name,
                            'group': group,
                            'dementia_label': dementia_label,
                            'file_path': str(cha_file),
                            'text_length': len(text),
                            'word_count': len(text.split()) if text else 0
                        })
                        
                        pitt_data.append(features)
                        
                    except Exception as e:
                        logger.warning(f"Error processing {cha_file}: {e}")
                        continue
        
        # Create DataFrame and save
        df = pd.DataFrame(pitt_data)
        
        if df.empty:
            logger.warning("No data extracted from Pitt Corpus")
            # Create empty CSV with expected columns
            empty_df = pd.DataFrame(columns=['participant_id', 'task_type', 'group', 'dementia_label'])
            empty_df.to_csv(output_file, index=False)
            return empty_df
            
        df.to_csv(output_file, index=False)
        
        logger.info(f"Extracted {len(df)} samples to {output_file}")
        logger.info(f"Groups: {df['group'].value_counts().to_dict()}")
        logger.info(f"Tasks: {df['task_type'].value_counts().to_dict()}")
        
        return df
    
    def _extract_participant_text(self, cha_file: Path) -> str:
        """Extract clean participant text from .cha file."""
        participant_lines = []
        
        try:
            with open(cha_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('*PAR:'):
                        # Extract participant content
                        content = line[5:].strip()  # Remove *PAR: prefix
                        
                        # Clean CHAT annotations
                        content = self._clean_chat_annotations(content)
                        
                        if content.strip():
                            participant_lines.append(content)
        
        except Exception as e:
            logger.warning(f"Failed to read {cha_file}: {e}")
        
        return ' '.join(participant_lines)
    
    def _clean_chat_annotations(self, text: str) -> str:
        """Clean CHAT format annotations."""
        import re
        
        # Remove timestamps
        text = re.sub(r'\d+_\d+', '', text)
        
        # Remove angle bracket annotations <...>
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove square bracket annotations [...] but keep content
        text = re.sub(r'\[([^\]]*)\]', r'\1', text)
        
        # Remove repetition markers [/] [//]
        text = re.sub(r'\[/+\]', '', text)
        
        # Remove filled pause markers &-
        text = re.sub(r'&-\w*', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_comprehensive_features(self, text: str, task_type: str) -> Dict:
        """Extract comprehensive features from text."""
        if not text:
            return self._get_zero_features()
        
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Basic linguistic features
        features = {
            # Text complexity
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(1, len(sentences)),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Lexical diversity
            'unique_word_ratio': len(set(words)) / max(1, len(words)),
            'lexical_diversity': len(set(words)),
            
            # Disfluency markers
            'hesitation_count': text.lower().count('um') + text.lower().count('uh') + text.lower().count('er'),
            'false_starts': text.count('I mean') + text.count('that is'),
            'self_corrections': text.count('no') + text.count('wait'),
            
            # Semantic markers
            'uncertainty_markers': text.lower().count('maybe') + text.lower().count('i think') + text.lower().count('probably'),
            'memory_references': text.lower().count('remember') + text.lower().count('forgot') + text.lower().count('memory'),
            
            # Task-specific features
            'task_completion_score': self._assess_task_completion(text, task_type),
            'narrative_coherence': self._assess_narrative_coherence(text, task_type),
            'semantic_fluency': self._assess_semantic_fluency(text, task_type),
        }
        
        # Derived cognitive features (with error handling)
        try:
            features.update({
                'cognitive_load_indicator': self._calculate_cognitive_load(features),
                'language_deterioration_score': self._calculate_deterioration_score(features),
                'discourse_coherence_score': self._calculate_discourse_coherence(text),
            })
        except Exception as e:
            # Fallback values if calculation fails
            features.update({
                'cognitive_load_indicator': 0.5,
                'language_deterioration_score': 0.5,
                'discourse_coherence_score': 0.5,
            })
        
        # Map to reminder system features
        reminder_features = self._map_to_reminder_features(features, text)
        features.update(reminder_features)
        
        return features
    
    def _get_zero_features(self) -> Dict:
        """Return zero-filled features for empty text."""
        return {
            # Basic features
            'word_count': 0, 'sentence_count': 0, 'avg_words_per_sentence': 0,
            'avg_word_length': 0, 'unique_word_ratio': 0, 'lexical_diversity': 0,
            'hesitation_count': 0, 'false_starts': 0, 'self_corrections': 0,
            'uncertainty_markers': 0, 'memory_references': 0, 'task_completion_score': 0,
            'narrative_coherence': 0, 'semantic_fluency': 0,
            
            # Derived features with safe defaults
            'cognitive_load_indicator': 0.5,
            'language_deterioration_score': 0.5,
            'discourse_coherence_score': 0.5
        }
    
    def _assess_task_completion(self, text: str, task_type: str) -> float:
        """Assess how well the task was completed."""
        text_lower = text.lower()
        
        if task_type == 'cookie':
            # Cookie theft picture description
            expected_elements = [
                'kitchen', 'boy', 'girl', 'cookie', 'jar', 'stool', 
                'water', 'sink', 'dishes', 'overflow', 'falling'
            ]
            found = sum(1 for elem in expected_elements if elem in text_lower)
            return found / len(expected_elements)
            
        elif task_type == 'fluency':
            # Word fluency task
            words = text.split()
            return min(1.0, len(words) / 20)  # Normalized by expected output
            
        elif task_type == 'recall':
            # Story recall task
            return min(1.0, len(text.split()) / 50)
            
        else:
            # General completion based on text length
            return min(1.0, len(text.split()) / 30)
    
    def _assess_narrative_coherence(self, text: str, task_type: str) -> float:
        """Assess narrative coherence."""
        if not text:
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence metrics
        coherence_score = 0.0
        
        # Check for logical connectives
        connectives = ['and', 'but', 'then', 'so', 'because', 'while', 'when']
        connective_count = sum(text.lower().count(conn) for conn in connectives)
        coherence_score += min(0.3, connective_count / len(sentences))
        
        # Check for temporal markers
        temporal_markers = ['first', 'then', 'next', 'finally', 'before', 'after']
        temporal_count = sum(text.lower().count(marker) for marker in temporal_markers)
        coherence_score += min(0.3, temporal_count / len(sentences))
        
        # Basic sentence flow (simple heuristic)
        coherence_score += 0.4  # Base score
        
        return min(1.0, coherence_score)
    
    def _assess_semantic_fluency(self, text: str, task_type: str) -> float:
        """Assess semantic fluency."""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        # Calculate semantic density (simplified)
        content_words = [w for w in words if len(w) > 3 and w.lower() not in 
                        ['the', 'and', 'that', 'this', 'with', 'they', 'them']]
        
        semantic_density = len(content_words) / len(words)
        
        # Penalize excessive repetition
        unique_ratio = len(set(words)) / len(words)
        
        fluency_score = (semantic_density + unique_ratio) / 2
        return min(1.0, fluency_score)
    
    def _calculate_cognitive_load(self, features: Dict) -> float:
        """Calculate cognitive load indicator."""
        try:
            load_score = 0.0
            
            # Higher hesitation = higher load
            word_count = features.get('word_count', 0)
            hesitation_count = features.get('hesitation_count', 0)
            
            if word_count > 0:
                hesitation_ratio = hesitation_count / word_count
                load_score += hesitation_ratio * 0.4
            
            # Lower diversity = higher load
            unique_word_ratio = features.get('unique_word_ratio', 0.5)
            load_score += (1 - unique_word_ratio) * 0.3
            
            # More uncertainty = higher load
            uncertainty_markers = features.get('uncertainty_markers', 0)
            if word_count > 0:
                uncertainty_ratio = uncertainty_markers / word_count
                load_score += uncertainty_ratio * 0.3
            
            return min(1.0, load_score)
        except Exception:
            return 0.5  # Default fallback value
    
    def _calculate_deterioration_score(self, features: Dict) -> float:
        """Calculate language deterioration score."""
        try:
            deterioration = 0.0
            
            # Short sentences may indicate deterioration
            avg_words_per_sentence = features.get('avg_words_per_sentence', 5)
            if avg_words_per_sentence < 5:
                deterioration += 0.3
            
            # Low lexical diversity
            unique_word_ratio = features.get('unique_word_ratio', 0.5)
            if unique_word_ratio < 0.5:
                deterioration += 0.3
            
            # High cognitive load
            cognitive_load = features.get('cognitive_load_indicator', 0.5)
            deterioration += cognitive_load * 0.4
            
            return min(1.0, deterioration)
        except Exception:
            return 0.5  # Default fallback value
    
    def _calculate_discourse_coherence(self, text: str) -> float:
        """Calculate discourse coherence."""
        if not text:
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence based on sentence connectivity
        coherence = 0.7  # Base coherence
        
        # Penalize very short responses
        if len(sentences) < 3:
            coherence -= 0.2
        
        # Reward appropriate length
        if 3 <= len(sentences) <= 8:
            coherence += 0.2
        
        return min(1.0, max(0.0, coherence))
    
    def _map_to_reminder_features(self, pitt_features: Dict, text: str) -> Dict:
        """Map Pitt features to reminder system feature names."""
        return {
            # Text features
            'hesitation_pauses': pitt_features['hesitation_count'],
            'semantic_incoherence': 1.0 - pitt_features['narrative_coherence'],
            'low_confidence_answers': pitt_features['uncertainty_markers'],
            'repeated_questions': 0,  # Not directly measurable
            'self_correction': pitt_features['self_corrections'],
            'response_coherence': pitt_features['discourse_coherence_score'],
            'word_finding_difficulty': pitt_features['hesitation_count'] / max(1, pitt_features['word_count']),
            'circumlocution': max(0, pitt_features['avg_words_per_sentence'] - 10) / 10,
            'tangentiality': 1.0 - pitt_features['task_completion_score'],
            
            # Temporal features (estimated)
            'response_time_seconds': np.random.uniform(10, 60),  # Would need audio timing
            'pause_frequency': pitt_features['hesitation_count'],
            'speech_rate': pitt_features['word_count'] / max(1, pitt_features['sentence_count']),
            'utterance_length': pitt_features['avg_words_per_sentence'],
            'pause_duration': np.random.uniform(0.5, 3.0),
            
            # Cognitive features
            'cognitive_risk_score': pitt_features['language_deterioration_score'],
            'confusion_detected': pitt_features['language_deterioration_score'] > 0.6,
            'memory_issue': pitt_features['memory_references'] > 0,
            'semantic_drift': 1.0 - pitt_features['semantic_fluency'],
            'discourse_coherence': pitt_features['discourse_coherence_score'],
            'lexical_diversity': pitt_features['unique_word_ratio'],
            
            # Context features
            'task_type_encoded': self._encode_task_type(text),
            'dementia_severity': pitt_features['language_deterioration_score'],
            
            # Pitt-specific features
            'pitt_dementia_markers': pitt_features['language_deterioration_score'],
            'narrative_coherence': pitt_features['narrative_coherence'],
            'task_completion': pitt_features['task_completion_score'],
            'linguistic_complexity': pitt_features['semantic_fluency'],
            'error_patterns': pitt_features['self_corrections'] + pitt_features['false_starts']
        }
    
    def _encode_task_type(self, text: str) -> int:
        """Encode task type based on content."""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ['cookie', 'kitchen', 'boy', 'jar']):
            return 0  # cookie task
        elif any(word in text_lower for word in ['animal', 'word', 'letter']):
            return 1  # fluency task
        elif any(word in text_lower for word in ['story', 'remember', 'told']):
            return 2  # recall task
        else:
            return 3  # other
    
    def create_balanced_dataset(self, pitt_file: str, synthetic_file: str, 
                              output_file: str, balance_ratio: float = 0.3) -> pd.DataFrame:
        """Create balanced dataset combining Pitt and synthetic data."""
        logger.info("Creating balanced dataset...")
        
        # Load datasets
        pitt_df = pd.read_csv(pitt_file)
        synthetic_df = pd.read_csv(synthetic_file)
        
        logger.info(f"Pitt samples: {len(pitt_df)}")
        logger.info(f"Synthetic samples: {len(synthetic_df)}")
        
        # Calculate balanced sample sizes
        pitt_size = len(pitt_df)
        synthetic_size = int(pitt_size * (1 - balance_ratio) / balance_ratio)
        
        if synthetic_size > len(synthetic_df):
            synthetic_size = len(synthetic_df)
            pitt_size = int(synthetic_size * balance_ratio / (1 - balance_ratio))
        
        logger.info(f"Using {pitt_size} Pitt samples and {synthetic_size} synthetic samples")
        
        # Sample datasets
        pitt_sample = pitt_df.sample(n=pitt_size, random_state=42)
        synthetic_sample = synthetic_df.sample(n=synthetic_size, random_state=42)
        
        # Add data source labels
        pitt_sample = pitt_sample.copy()
        synthetic_sample = synthetic_sample.copy()
        pitt_sample['data_source'] = 'pitt_corpus'
        synthetic_sample['data_source'] = 'synthetic'
        
        # Align columns
        common_columns = set(pitt_sample.columns) & set(synthetic_sample.columns)
        
        # Add missing columns with defaults
        for col in pitt_sample.columns:
            if col not in synthetic_sample.columns:
                synthetic_sample[col] = self._get_default_for_column(col, len(synthetic_sample))
        
        for col in synthetic_sample.columns:
            if col not in pitt_sample.columns:
                pitt_sample[col] = self._get_default_for_column(col, len(pitt_sample))
        
        # Combine datasets
        combined_df = pd.concat([pitt_sample, synthetic_sample], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Balanced dataset saved to {output_file}")
        logger.info(f"Total samples: {len(combined_df)}")
        logger.info(f"Data sources: {combined_df['data_source'].value_counts().to_dict()}")
        
        return combined_df
    
    def _get_default_for_column(self, col: str, size: int):
        """Get appropriate default value for column."""
        if 'encoded' in col or col.endswith('_score') or col.endswith('_ratio'):
            return np.zeros(size)
        elif 'detected' in col or col.endswith('_issue') or col.endswith('_alert'):
            return np.zeros(size, dtype=bool)
        elif 'time' in col or 'duration' in col:
            return np.random.uniform(5, 30, size)
        elif 'count' in col or col.endswith('_markers'):
            return np.zeros(size, dtype=int)
        else:
            return [''] * size if col in ['reminder_text', 'user_response'] else np.zeros(size)
    
    def validate_integration(self, integrated_file: str) -> Dict:
        """Validate the integrated dataset."""
        logger.info("Validating integrated dataset...")
        
        df = pd.read_csv(integrated_file)
        
        validation_report = {
            'total_samples': len(df),
            'data_sources': df['data_source'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_ranges': {},
            'data_quality_issues': []
        }
        
        # Check feature ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            validation_report['feature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        # Check for data quality issues
        for col in numeric_columns:
            if df[col].isnull().sum() > len(df) * 0.1:
                validation_report['data_quality_issues'].append(f"High missing values in {col}")
            
            if df[col].std() == 0:
                validation_report['data_quality_issues'].append(f"No variance in {col}")
        
        logger.info("Validation complete")
        logger.info(f"Quality issues found: {len(validation_report['data_quality_issues'])}")
        
        return validation_report


def main():
    """Main function for data integration."""
    parser = argparse.ArgumentParser(description="Integrate Pitt Corpus with synthetic reminder data")
    parser.add_argument("--extract-pitt", action="store_true", help="Extract features from Pitt Corpus")
    parser.add_argument("--create-balanced", action="store_true", help="Create balanced integrated dataset")
    parser.add_argument("--validate", action="store_true", help="Validate integrated dataset")
    parser.add_argument("--output", default="data/integrated_training_data.csv", help="Output file path")
    parser.add_argument("--pitt-features", default="data/pitt_features.csv", help="Pitt features file")
    parser.add_argument("--synthetic-file", default="data/synthetic_reminder_data.csv", help="Synthetic data file")
    parser.add_argument("--balance-ratio", type=float, default=0.3, help="Ratio of Pitt to total data")
    
    args = parser.parse_args()
    
    integrator = PittDataIntegrator()
    
    if args.extract_pitt:
        logger.info("Extracting Pitt Corpus features...")
        integrator.extract_pitt_features(args.pitt_features)
    
    if args.create_balanced:
        logger.info("Creating balanced dataset...")
        integrator.create_balanced_dataset(
            args.pitt_features, args.synthetic_file, args.output, args.balance_ratio
        )
    
    if args.validate:
        logger.info("Validating dataset...")
        report = integrator.validate_integration(args.output)
        
        # Save validation report
        report_file = args.output.replace('.csv', '_validation.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_file}")
    
    logger.info("Data integration completed!")


if __name__ == "__main__":
    main()