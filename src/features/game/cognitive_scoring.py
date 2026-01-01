import numpy as np
from typing import Dict, List, Tuple

class CognitiveScorer:
    """
    Implements cognitive scoring algorithms:
    - Speed-Accuracy Composite (SAC)
    - Inverse Efficiency Score (IES)
    - Motor-Adjusted Reaction Time
    """
    
    def __init__(self, motor_baseline: float = None):
        """
        Initialize scorer with optional motor baseline
        
        Args:
            motor_baseline: User's baseline motor reaction time in ms
        """
        self.motor_baseline = motor_baseline or 0
    
    def calculate_accuracy_rate(self, correct: int, total: int) -> float:
        """
        Calculate accuracy rate
        
        Args:
            correct: Number of correct responses
            total: Total number of attempts
            
        Returns:
            Accuracy rate (0-1)
        """
        if total == 0:
            return 0.0
        return correct / total
    
    def calculate_error_rate(self, errors: int, total: int) -> float:
        """
        Calculate error rate
        
        Args:
            errors: Number of errors
            total: Total number of attempts
            
        Returns:
            Error rate (0-1)
        """
        if total == 0:
            return 0.0
        return errors / total
    
    def calculate_avg_reaction_time(self, reaction_times: List[float]) -> float:
        """
        Calculate average reaction time
        
        Args:
            reaction_times: List of reaction times in ms
            
        Returns:
            Average reaction time in ms
        """
        if not reaction_times:
            return 0.0
        return np.mean(reaction_times)
    
    def calculate_motor_adjusted_rt(self, raw_rt: float) -> float:
        """
        Adjust reaction time for motor baseline
        
        Formula: RT_adj = RT_raw - motor_baseline
        
        Args:
            raw_rt: Raw reaction time in ms
            
        Returns:
            Motor-adjusted reaction time in ms
        """
        adjusted = raw_rt - self.motor_baseline
        return max(adjusted, 0)  # Ensure non-negative
    
    def calculate_sac_score(self, accuracy: float, avg_rt: float, 
                           alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        Calculate Speed-Accuracy Composite (SAC)
        
        Formula: SAC = α × Accuracy - β × (RT_normalized)
        
        Args:
            accuracy: Accuracy rate (0-1)
            avg_rt: Average reaction time in ms
            alpha: Weight for accuracy (default: 0.5)
            beta: Weight for reaction time (default: 0.5)
            
        Returns:
            SAC score (higher is better)
        """
        # Normalize RT to 0-1 scale (assuming max RT = 10000ms)
        rt_normalized = min(avg_rt / 10000.0, 1.0)
        
        sac = (alpha * accuracy) - (beta * rt_normalized)
        return sac
    
    def calculate_ies_score(self, accuracy: float, avg_rt: float) -> float:
        """
        Calculate Inverse Efficiency Score (IES)
        
        Formula: IES = RT_adj / Accuracy
        (Lower IES = better cognitive efficiency)
        
        Args:
            accuracy: Accuracy rate (0-1)
            avg_rt: Average reaction time in ms
            
        Returns:
            IES score (lower is better)
        """
        if accuracy == 0:
            return float('inf')
        
        # Use motor-adjusted RT if baseline is available
        rt_adjusted = self.calculate_motor_adjusted_rt(avg_rt)
        ies = rt_adjusted / accuracy
        return ies
    
    def calculate_session_scores(self, session_data: Dict) -> Dict:
        """
        Calculate all cognitive scores for a session
        
        Args:
            session_data: Dictionary containing:
                - total_matches: int
                - total_errors: int
                - total_attempts: int
                - reaction_times: List[float]
                - hints_used: int
                
        Returns:
            Dictionary with all calculated scores
        """
        total_attempts = session_data.get('total_attempts', 0)
        total_matches = session_data.get('total_matches', 0)
        total_errors = session_data.get('total_errors', 0)
        reaction_times = session_data.get('reaction_times', [])
        
        # Calculate basic metrics
        accuracy = self.calculate_accuracy_rate(total_matches, total_attempts)
        error_rate = self.calculate_error_rate(total_errors, total_attempts)
        avg_rt = self.calculate_avg_reaction_time(reaction_times)
        motor_adj_rt = self.calculate_motor_adjusted_rt(avg_rt)
        
        # Calculate composite scores
        sac = self.calculate_sac_score(accuracy, motor_adj_rt)
        ies = self.calculate_ies_score(accuracy, motor_adj_rt)
        
        # Calculate RT consistency (standard deviation)
        rt_std = np.std(reaction_times) if len(reaction_times) > 1 else 0.0
        
        return {
            'accuracy_rate': round(accuracy, 4),
            'error_rate': round(error_rate, 4),
            'avg_reaction_time': round(avg_rt, 2),
            'motor_adjusted_rt': round(motor_adj_rt, 2),
            'sac_score': round(sac, 4),
            'ies_score': round(ies, 2),
            'rt_std': round(rt_std, 2),
            'rt_consistency': round(1 - (rt_std / avg_rt), 4) if avg_rt > 0 else 0
        }
    
    def calculate_temporal_features(self, sessions_scores: List[Dict]) -> Dict:
        """
        Calculate temporal features across multiple sessions
        
        Args:
            sessions_scores: List of session score dictionaries
            
        Returns:
            Dictionary with temporal trend features
        """
        if not sessions_scores:
            return {}
        
        # Extract score sequences
        sac_scores = [s['sac_score'] for s in sessions_scores]
        accuracy_scores = [s['accuracy_rate'] for s in sessions_scores]
        rt_scores = [s['motor_adjusted_rt'] for s in sessions_scores]
        
        # Calculate trends (linear regression slope)
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]  # slope
        
        sac_trend = calculate_trend(sac_scores)
        accuracy_trend = calculate_trend(accuracy_scores)
        rt_trend = calculate_trend(rt_scores)
        
        # Calculate variability
        sac_variability = np.std(sac_scores) if len(sac_scores) > 1 else 0
        
        return {
            'avg_sac': round(np.mean(sac_scores), 4),
            'sac_trend': round(sac_trend, 6),
            'sac_variability': round(sac_variability, 4),
            'avg_accuracy': round(np.mean(accuracy_scores), 4),
            'accuracy_trend': round(accuracy_trend, 6),
            'avg_rt': round(np.mean(rt_scores), 2),
            'rt_trend': round(rt_trend, 4),
            'num_sessions': len(sessions_scores)
        }
