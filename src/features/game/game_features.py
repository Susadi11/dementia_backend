import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime

class GameFeatureExtractor:
    """
    Extract cognitive features from raw gameplay data
    Prepares features for LSTM and risk classification models
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_from_actions(self, actions: List[Dict]) -> Dict:
        """
        Extract features from a list of game actions
        
        Args:
            actions: List of action dictionaries with keys:
                - action_type: 'flip', 'match', 'error', 'hint'
                - timestamp: datetime
                - reaction_time_ms: int
                - is_correct: bool
                
        Returns:
            Dictionary of extracted features
        """
        if not actions:
            return self._get_empty_features()
        
        # Separate by action type
        flips = [a for a in actions if a['action_type'] == 'flip']
        matches = [a for a in actions if a['action_type'] == 'match' and a.get('is_correct')]
        errors = [a for a in actions if a['action_type'] == 'error' or 
                  (a['action_type'] == 'match' and not a.get('is_correct'))]
        hints = [a for a in actions if a['action_type'] == 'hint']
        
        # Extract reaction times
        all_rts = [a['reaction_time_ms'] for a in actions if a.get('reaction_time_ms')]
        match_rts = [a['reaction_time_ms'] for a in matches if a.get('reaction_time_ms')]
        
        # Calculate temporal features
        session_duration = self._calculate_duration(actions)
        action_pace = len(actions) / (session_duration / 60) if session_duration > 0 else 0
        
        # Calculate hesitation patterns
        hesitations = self._detect_hesitations(all_rts)
        
        features = {
            # Count features
            'total_actions': len(actions),
            'total_flips': len(flips),
            'total_matches': len(matches),
            'total_errors': len(errors),
            'hints_used': len(hints),
            
            # Rate features
            'match_rate': len(matches) / len(flips) if len(flips) > 0 else 0,
            'error_rate': len(errors) / len(flips) if len(flips) > 0 else 0,
            'hint_rate': len(hints) / len(flips) if len(flips) > 0 else 0,
            
            # Reaction time features
            'avg_rt': np.mean(all_rts) if all_rts else 0,
            'median_rt': np.median(all_rts) if all_rts else 0,
            'std_rt': np.std(all_rts) if len(all_rts) > 1 else 0,
            'min_rt': np.min(all_rts) if all_rts else 0,
            'max_rt': np.max(all_rts) if all_rts else 0,
            
            # Match-specific RT
            'avg_match_rt': np.mean(match_rts) if match_rts else 0,
            'std_match_rt': np.std(match_rts) if len(match_rts) > 1 else 0,
            
            # Temporal features
            'session_duration_sec': session_duration,
            'action_pace_per_min': action_pace,
            
            # Cognitive load indicators
            'hesitation_count': hesitations['count'],
            'avg_hesitation_duration': hesitations['avg_duration'],
            'rt_variability': np.std(all_rts) / np.mean(all_rts) if all_rts and np.mean(all_rts) > 0 else 0,
            
            # Performance consistency
            'early_accuracy': self._calculate_segment_accuracy(matches, errors, segment='early'),
            'late_accuracy': self._calculate_segment_accuracy(matches, errors, segment='late'),
            'accuracy_decline': 0  # Will be calculated
        }
        
        # Calculate accuracy decline
        features['accuracy_decline'] = features['early_accuracy'] - features['late_accuracy']
        
        return features
    
    def _calculate_duration(self, actions: List[Dict]) -> float:
        """Calculate session duration in seconds"""
        if len(actions) < 2:
            return 0.0
        
        timestamps = [a['timestamp'] for a in actions if 'timestamp' in a]
        if not timestamps:
            return 0.0
        
        start = min(timestamps)
        end = max(timestamps)
        
        # Handle both datetime objects and strings
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        
        duration = (end - start).total_seconds()
        return duration
    
    def _detect_hesitations(self, reaction_times: List[float], threshold: float = 5000) -> Dict:
        """
        Detect hesitations (unusually long reaction times)
        
        Args:
            reaction_times: List of RTs in ms
            threshold: RT threshold for hesitation (default: 5000ms)
        """
        hesitations = [rt for rt in reaction_times if rt > threshold]
        
        return {
            'count': len(hesitations),
            'avg_duration': np.mean(hesitations) if hesitations else 0
        }
    
    def _calculate_segment_accuracy(self, matches: List, errors: List, 
                                   segment: str = 'early') -> float:
        """
        Calculate accuracy for early or late segment of session
        
        Args:
            matches: List of match actions
            errors: List of error actions
            segment: 'early' or 'late'
        """
        all_attempts = matches + errors
        if not all_attempts:
            return 0.0
        
        # Sort by timestamp if available
        if all_attempts and 'timestamp' in all_attempts[0]:
            all_attempts = sorted(all_attempts, key=lambda x: x['timestamp'])
        
        # Split into segments
        mid_point = len(all_attempts) // 2
        
        if segment == 'early':
            segment_attempts = all_attempts[:mid_point] if mid_point > 0 else all_attempts
        else:  # late
            segment_attempts = all_attempts[mid_point:] if mid_point > 0 else all_attempts
        
        if not segment_attempts:
            return 0.0
        
        segment_matches = [a for a in segment_attempts if a in matches]
        return len(segment_matches) / len(segment_attempts)
    
    def _get_empty_features(self) -> Dict:
        """Return zero-filled feature dictionary"""
        return {
            'total_actions': 0, 'total_flips': 0, 'total_matches': 0,
            'total_errors': 0, 'hints_used': 0, 'match_rate': 0,
            'error_rate': 0, 'hint_rate': 0, 'avg_rt': 0,
            'median_rt': 0, 'std_rt': 0, 'min_rt': 0, 'max_rt': 0,
            'avg_match_rt': 0, 'std_match_rt': 0, 'session_duration_sec': 0,
            'action_pace_per_min': 0, 'hesitation_count': 0,
            'avg_hesitation_duration': 0, 'rt_variability': 0,
            'early_accuracy': 0, 'late_accuracy': 0, 'accuracy_decline': 0
        }
    
    def prepare_lstm_sequence(self, session_features: List[Dict], 
                            sequence_length: int = 10) -> np.ndarray:
        """
        Prepare feature sequences for LSTM input
        
        Args:
            session_features: List of feature dictionaries from multiple sessions
            sequence_length: Number of sessions to include in sequence
            
        Returns:
            numpy array of shape (n_sequences, sequence_length, n_features)
        """
        if not session_features:
            return np.array([])
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(session_features)
        
        # Select key features for LSTM
        feature_cols = [
            'match_rate', 'error_rate', 'avg_rt', 'std_rt',
            'rt_variability', 'hesitation_count', 'accuracy_decline',
            'action_pace_per_min', 'hints_used'
        ]
        
        # Ensure all columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        feature_data = df[feature_cols].values
        
        # Create sequences
        sequences = []
        for i in range(len(feature_data) - sequence_length + 1):
            seq = feature_data[i:i + sequence_length]
            sequences.append(seq)
        
        if not sequences:
            # If not enough data for full sequence, pad with zeros
            seq = np.pad(feature_data, 
                        ((sequence_length - len(feature_data), 0), (0, 0)),
                        mode='constant')
            sequences = [seq]
        
        return np.array(sequences)
    
    def extract_risk_features(self, temporal_features: Dict, 
                            cognitive_scores: List[Dict]) -> Dict:
        """
        Extract aggregated features for risk classification
        
        Args:
            temporal_features: Features from temporal analysis
            cognitive_scores: List of cognitive score dictionaries
            
        Returns:
            Feature dictionary for risk classifier
        """
        if not cognitive_scores:
            return {}
        
        # Calculate aggregated metrics
        recent_sessions = cognitive_scores[-5:] if len(cognitive_scores) >= 5 else cognitive_scores
        
        sac_scores = [s['sac_score'] for s in recent_sessions]
        accuracy_scores = [s['accuracy_rate'] for s in recent_sessions]
        ies_scores = [s['ies_score'] for s in recent_sessions]
        
        risk_features = {
            'avg_sac': np.mean(sac_scores),
            'min_sac': np.min(sac_scores),
            'sac_trend': temporal_features.get('sac_trend', 0),
            'sac_variability': np.std(sac_scores),
            
            'avg_accuracy': np.mean(accuracy_scores),
            'min_accuracy': np.min(accuracy_scores),
            'accuracy_trend': temporal_features.get('accuracy_trend', 0),
            
            'avg_ies': np.mean(ies_scores),
            'max_ies': np.max(ies_scores),
            
            'num_sessions': temporal_features.get('num_sessions', len(cognitive_scores)),
            'lstm_decline_score': 0  # Will be filled by LSTM prediction
        }
        
        return risk_features


# Example usage
if __name__ == "__main__":
    extractor = GameFeatureExtractor()
    
    # Sample actions data
    sample_actions = [
        {'action_type': 'flip', 'timestamp': datetime.now(), 'reaction_time_ms': 2300},
        {'action_type': 'flip', 'timestamp': datetime.now(), 'reaction_time_ms': 2400},
        {'action_type': 'match', 'timestamp': datetime.now(), 'reaction_time_ms': 2500, 'is_correct': True},
        {'action_type': 'flip', 'timestamp': datetime.now(), 'reaction_time_ms': 2200},
        {'action_type': 'error', 'timestamp': datetime.now(), 'reaction_time_ms': 3100},
    ]
    
    features = extractor.extract_from_actions(sample_actions)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")