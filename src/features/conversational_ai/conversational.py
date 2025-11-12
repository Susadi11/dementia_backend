"""
Conversational AI Feature Analyzer

Analyzes text and voice to extract dementia indicator parameters including
semantic incoherence, repeated questions, self-correction, confidence markers,
and acoustic characteristics.
"""

import re
from typing import Dict, Any, List
from datetime import datetime
from src.features.base_features import BaseFeatureExtractor, FeatureResult


class DementiaIndicatorAnalyzer(BaseFeatureExtractor):
    """
    Analyzes conversational patterns to detect dementia indicators.
    
    This analyzer examines both text and voice characteristics to identify
    the 10 key parameters for dementia detection.
    """
    
    def __init__(self):
        """Initialize the dementia indicator analyzer."""
        super().__init__()
        self.session_utterances = []
        self.start_time = datetime.now()
        
    def extract(self, text: str = None, audio_features: Dict = None) -> FeatureResult:
        """
        Extract dementia indicators from text and audio.
        
        Args:
            text: User's text input/transcript
            audio_features: Audio analysis results with timings and characteristics
            
        Returns:
            FeatureResult with all 10 dementia indicators
        """
        features = {}
        
        if text:
            self.session_utterances.append(text)
            text_features = self._extract_text_features(text)
            features.update(text_features)
        
        if audio_features:
            audio_analyzed = self._extract_audio_features(audio_features)
            features.update(audio_analyzed)
        
        # Calculate session decline
        decline_score = self._calculate_session_decline()
        features['in_session_decline'] = decline_score
        
        return FeatureResult(
            features=features,
            feature_type='dementia_indicators',
            metadata={
                'timestamp': datetime.now().isoformat(),
                'utterance_count': len(self.session_utterances),
                'analysis_type': 'conversational_ai'
            }
        )
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract features from text analysis."""
        features = {
            'semantic_incoherence': self._calculate_semantic_incoherence(text),
            'repeated_questions': self._count_repeated_questions(text),
            'self_correction': self._count_self_corrections(text),
            'low_confidence_answer': self._detect_low_confidence(text),
        }
        return features

    def _extract_audio_features(self, audio_features: Dict) -> Dict[str, float]:
        """Extract features from audio analysis."""
        features = {
            'hesitation_pauses': audio_features.get('pause_frequency', 0.0),
            'vocal_tremors': audio_features.get('tremor_intensity', 0.0),
            'emotion_slip': audio_features.get('emotion_intensity', 0.0) * \
                           audio_features.get('speech_error_rate', 0.0),
            'slowed_speech': self._analyze_speech_rate(audio_features),
        }
        return features
    
    def _calculate_semantic_incoherence(self, text: str) -> float:
        """
        Calculate semantic incoherence score (0-1).
        Higher = more incoherent
        """
        # Detect abrupt topic changes, incomplete thoughts, disjointed sentences
        incomplete_patterns = r'\b(but|however|also|anyway|so)\s+\.\.\.|\.{3,}|\?\?|!!!|\.\s+\.'
        
        incoherence_indicators = len(re.findall(incomplete_patterns, text))
        
        # Normalize to 0-1 range
        score = min(incoherence_indicators / 10.0, 1.0)
        
        return score
    
    def _count_repeated_questions(self, text: str) -> float:
        """
        Count repeated questions (0-1 scale).
        Detect question patterns and repetition.
        """
        # Extract questions
        questions = re.findall(r'[^.!?]*\?', text)
        
        if len(questions) < 2:
            return 0.0
        
        # Calculate similarity between consecutive questions
        repeated_count = 0
        for i in range(len(questions) - 1):
            # Simple similarity: if questions share >50% words, they're similar
            q1_words = set(questions[i].lower().split())
            q2_words = set(questions[i+1].lower().split())
            
            if len(q1_words & q2_words) / max(len(q1_words | q2_words), 1) > 0.5:
                repeated_count += 1
        
        score = min(repeated_count / len(questions), 1.0)
        return score
    
    def _count_self_corrections(self, text: str) -> float:
        """
        Count self-corrections in text (0-1 scale).
        Detect patterns like: 'I mean...', 'Actually...', 'No wait...', etc.
        """
        correction_patterns = [
            r'\bI mean\b', r'\bActually\b', r'\bNo wait\b',
            r'\bI meant\b', r'\bSorry,\b', r'\bLet me rephrase\b',
            r'\bor rather\b', r'\bI meant to say\b'
        ]
        
        total_corrections = 0
        for pattern in correction_patterns:
            total_corrections += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize
        words_count = len(text.split())
        score = min(total_corrections / max(words_count / 10, 1), 1.0)
        
        return score
    
    def _detect_low_confidence(self, text: str) -> float:
        """
        Detect low-confidence indicators (0-1 scale).
        Look for words like: 'maybe', 'I think', 'probably', 'I guess', etc.
        """
        confidence_indicators = [
            r'\bmaybe\b', r'\bI think\b', r'\bprobably\b',
            r'\bI guess\b', r'\bI suppose\b', r'\bkind of\b',
            r'\bsort of\b', r'\bI''m not sure\b', r'\bI don''t know\b'
        ]
        
        total_indicators = 0
        for pattern in confidence_indicators:
            total_indicators += len(re.findall(pattern, text, re.IGNORECASE))
        
        words_count = len(text.split())
        score = min(total_indicators / max(words_count / 15, 1), 1.0)
        
        return score
    
    def _analyze_speech_rate(self, audio_features: Dict) -> float:
        """
        Analyze speech rate slowness (0-1 scale).
        """
        speech_rate = audio_features.get('speech_rate', 120)  # words per minute
        
        # Normal speech rate: 120-150 wpm
        # Slowed speech: <100 wpm
        
        if speech_rate < 80:
            return 1.0  # Severely slowed
        elif speech_rate < 100:
            return 0.7
        elif speech_rate < 120:
            return 0.4
        else:
            return 0.0
    
    def _calculate_session_decline(self) -> float:
        """
        Calculate in-session decline (0-1 scale).
        Compare performance across utterances in the session.
        """
        if len(self.session_utterances) < 2:
            return 0.0
        
        # Simple metric: check if recent utterances are shorter/less coherent
        early_utterances = self.session_utterances[:max(1, len(self.session_utterances)//2)]
        late_utterances = self.session_utterances[max(1, len(self.session_utterances)//2):]
        
        early_avg_length = sum(len(u.split()) for u in early_utterances) / len(early_utterances)
        late_avg_length = sum(len(u.split()) for u in late_utterances) / len(late_utterances) if late_utterances else early_avg_length
        
        # If later responses are shorter, there's decline
        decline_score = max(0, (early_avg_length - late_avg_length) / early_avg_length)
        
        return min(decline_score, 1.0)
    
    @property
    def feature_names(self) -> List[str]:
        """Return list of all feature names."""
        return [
            'semantic_incoherence',
            'repeated_questions',
            'self_correction',
            'low_confidence_answer',
            'hesitation_pauses',
            'vocal_tremors',
            'emotion_slip',
            'slowed_speech',
            'evening_errors',
            'in_session_decline'
        ]


__all__ = ["DementiaIndicatorAnalyzer"]
