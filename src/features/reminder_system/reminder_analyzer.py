"""
Pitt Corpus-Based Reminder Analyzer

Analyzes user responses to reminders using models trained on 
the DementiaBank Pitt Corpus dataset. Detects cognitive patterns,
confusion, and memory issues in reminder interactions.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

from src.features.conversational_ai.feature_extractor import FeatureExtractor
from src.features.conversational_ai.nlp.nlp_engine import NLPEngine
from src.utils.helpers import calculate_overall_risk
from .reminder_models import InteractionType

logger = logging.getLogger(__name__)


class PittBasedReminderAnalyzer:
    """
    Analyzes reminder responses using Pitt Corpus trained models.
    
    Uses BERT-based NLP and dementia indicators extracted from
    real dementia patient speech patterns to assess user responses.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize analyzer with feature extraction and NLP engines.
        
        Args:
            model_path: Optional path to Pitt-trained model for predictions
        """
        self.feature_extractor = FeatureExtractor()
        self.nlp_engine = NLPEngine()
        self.model_path = model_path
        self.model = None
        
        # Load model if path provided and exists
        if model_path and Path(model_path).exists():
            try:
                from src.models.conversational_ai.text_model_trainer import TextModelTrainer
                self.model = TextModelTrainer()
                self.model.load_model(model_path)
                logger.info(f"Loaded Pitt-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                self.model = None
    
    def analyze_reminder_response(
        self,
        user_response: str,
        audio_path: Optional[str] = None,
        reminder_context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze user's response to a reminder notification.
        
        Examples of responses and what they indicate:
        - "I already did it" → Clear confirmation
        - "Um... I think I did... maybe?" → Confusion/uncertainty
        - "What medicine?" → Repeated question/memory issue
        - "Later... not now" → Deliberate delay
        - "I don't remember" → Memory impairment
        
        Args:
            user_response: User's text response to reminder
            audio_path: Optional path to audio recording of response
            reminder_context: Optional context about the reminder
        
        Returns:
            Dict containing:
                - cognitive_risk_score: 0-1 score of cognitive impairment
                - confusion_detected: Whether user seems confused
                - memory_issue_detected: Whether memory problem detected
                - uncertainty_detected: Whether user is uncertain
                - interaction_type: Classified interaction type
                - recommended_action: What to do next
                - caregiver_alert_needed: Whether to notify caregiver
                - features: Extracted dementia indicators
                - confidence: Model confidence in assessment
        """
        try:
            # Extract features from response
            features = self.feature_extractor.extract_features_normalized(
                transcript_text=user_response,
                audio_path=audio_path
            )
            
            # Calculate cognitive risk score
            cognitive_risk = calculate_overall_risk(features)
            
            # Analyze with NLP engine for additional context
            nlp_analysis = self.nlp_engine.analyze(user_response)
            
            # Detect specific patterns
            confusion_detected = self._detect_confusion(features, user_response)
            memory_issue = self._detect_memory_issue(features, user_response)
            uncertainty = self._detect_uncertainty(features, user_response)
            
            # Classify interaction type
            interaction_type = self._classify_interaction(
                user_response, features, confusion_detected, memory_issue
            )
            
            # Get model prediction if available
            model_confidence = 0.0
            if self.model:
                try:
                    feature_values = list(features.values())
                    prediction, probabilities = self.model.predict([feature_values])
                    cognitive_risk = max(cognitive_risk, probabilities[0][1])
                    model_confidence = float(max(probabilities[0]))
                except Exception as e:
                    logger.warning(f"Model prediction failed: {e}")
            
            # Determine recommended action
            action = self._determine_action(
                cognitive_risk, confusion_detected, memory_issue, 
                uncertainty, interaction_type, reminder_context
            )
            
            # Decide if caregiver alert needed
            caregiver_alert = self._should_alert_caregiver(
                cognitive_risk, confusion_detected, memory_issue,
                interaction_type, reminder_context
            )
            
            return {
                'cognitive_risk_score': float(cognitive_risk),
                'confusion_detected': confusion_detected,
                'memory_issue_detected': memory_issue,
                'uncertainty_detected': uncertainty,
                'interaction_type': interaction_type.value,
                'recommended_action': action,
                'caregiver_alert_needed': caregiver_alert,
                'features': features,
                'confidence': model_confidence,
                'nlp_analysis': nlp_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing reminder response: {e}", exc_info=True)
            return {
                'cognitive_risk_score': 0.5,
                'confusion_detected': False,
                'memory_issue_detected': False,
                'uncertainty_detected': False,
                'interaction_type': InteractionType.IGNORED.value,
                'recommended_action': 'error_occurred_retry',
                'caregiver_alert_needed': False,
                'features': {},
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _detect_confusion(self, features: Dict, response_text: str) -> bool:
        """Detect if user is confused about the reminder."""
        # High semantic incoherence indicates confusion
        if features.get('semantic_incoherence', 0) > 0.4:
            return True
        
        # Check for confusion keywords
        confusion_keywords = [
            'what', 'which', 'confused', 'don\'t understand',
            'what do you mean', 'huh', 'what are you talking about'
        ]
        response_lower = response_text.lower()
        if any(keyword in response_lower for keyword in confusion_keywords):
            return True
        
        return False
    
    def _detect_memory_issue(self, features: Dict, response_text: str) -> bool:
        """Detect if user has memory issues."""
        # Repeated questions indicate memory problems
        if features.get('repeated_questions', 0) > 0:
            return True
        
        # Check for memory-related phrases
        memory_keywords = [
            'don\'t remember', 'forgot', 'can\'t recall',
            'did i already', 'did i do', 'have i done'
        ]
        response_lower = response_text.lower()
        if any(keyword in response_lower for keyword in memory_keywords):
            return True
        
        return False
    
    def _detect_uncertainty(self, features: Dict, response_text: str) -> bool:
        """Detect if user is uncertain."""
        # Low confidence answers and hesitation pauses
        if features.get('low_confidence_answers', 0) > 0.5:
            return True
        
        if features.get('hesitation_pauses', 0) > 3:
            return True
        
        # Check for uncertainty words
        uncertainty_keywords = [
            'maybe', 'i think', 'probably', 'not sure',
            'might have', 'possibly', 'perhaps'
        ]
        response_lower = response_text.lower()
        if any(keyword in response_lower for keyword in uncertainty_keywords):
            return True
        
        return False
    
    def _classify_interaction(
        self,
        response_text: str,
        features: Dict,
        confusion: bool,
        memory_issue: bool
    ) -> InteractionType:
        """Classify the type of interaction."""
        response_lower = response_text.lower()
        
        # Confusion takes priority
        if confusion:
            return InteractionType.CONFUSED
        
        # Memory issue
        if memory_issue:
            return InteractionType.REPEATED_QUESTION
        
        # Confirmed completion
        confirmed_keywords = [
            'yes', 'done', 'completed', 'finished', 'already did',
            'took it', 'took my', 'i did', 'just did'
        ]
        if any(keyword in response_lower for keyword in confirmed_keywords):
            return InteractionType.CONFIRMED
        
        # Deliberate delay
        delay_keywords = [
            'later', 'wait', 'not now', 'busy', 'in a minute',
            'soon', 'after', 'remind me again'
        ]
        if any(keyword in response_lower for keyword in delay_keywords):
            return InteractionType.DELAYED
        
        # Partial completion
        partial_keywords = ['started', 'working on', 'about to', 'almost']
        if any(keyword in response_lower for keyword in partial_keywords):
            return InteractionType.PARTIAL_COMPLETION
        
        # Default to ignored if no clear response
        return InteractionType.IGNORED
    
    def _determine_action(
        self,
        cognitive_risk: float,
        confusion: bool,
        memory_issue: bool,
        uncertainty: bool,
        interaction_type: InteractionType,
        reminder_context: Optional[Dict]
    ) -> str:
        """Determine the recommended action based on analysis."""
        # Critical cognitive risk - escalate immediately
        if cognitive_risk > 0.7:
            return 'escalate_to_caregiver_urgent'
        
        # Confusion detected - simplify and provide context
        if confusion:
            return 'simplify_reminder_with_context'
        
        # Memory issue - provide detailed context and repeat
        if memory_issue:
            return 'provide_context_and_repeat'
        
        # High uncertainty - offer more information
        if uncertainty and cognitive_risk > 0.5:
            return 'offer_additional_help'
        
        # Confirmed - mark as complete
        if interaction_type == InteractionType.CONFIRMED:
            return 'mark_completed'
        
        # Delayed - reschedule
        if interaction_type == InteractionType.DELAYED:
            return 'snooze_reminder'
        
        # Partial completion - follow up
        if interaction_type == InteractionType.PARTIAL_COMPLETION:
            return 'schedule_follow_up'
        
        # Moderate cognitive risk - increase monitoring
        if cognitive_risk > 0.6:
            return 'increase_monitoring'
        
        # Default - normal processing
        return 'continue_normal'
    
    def _should_alert_caregiver(
        self,
        cognitive_risk: float,
        confusion: bool,
        memory_issue: bool,
        interaction_type: InteractionType,
        reminder_context: Optional[Dict]
    ) -> bool:
        """Determine if caregiver should be alerted."""
        # High cognitive risk
        if cognitive_risk > 0.7:
            return True
        
        # Confusion or memory issues detected
        if confusion or memory_issue:
            return True
        
        # Check if reminder is critical (medication, appointment)
        if reminder_context:
            priority = reminder_context.get('priority', 'medium')
            category = reminder_context.get('category', 'general')
            
            # Critical reminders with any issues
            if priority == 'critical' and cognitive_risk > 0.5:
                return True
            
            # Medication reminders with confusion
            if category == 'medication' and (confusion or cognitive_risk > 0.6):
                return True
        
        return False


def test_reminder_analyzer():
    """Test the reminder analyzer with sample responses."""
    
    analyzer = PittBasedReminderAnalyzer()
    
    test_cases = [
        {
            'response': "Yes, I took my tablets after breakfast",
            'description': "Clear confirmation"
        },
        {
            'response': "Um... I think I already did that... or maybe not?",
            'description': "Confusion and uncertainty"
        },
        {
            'response': "What medicine? What are you talking about?",
            'description': "Memory issue and confusion"
        },
        {
            'response': "Later... I'm busy right now",
            'description': "Deliberate delay"
        },
        {
            'response': "I don't remember... did I already do it?",
            'description': "Memory and uncertainty"
        },
        {
            'response': "Just finished taking it, thank you",
            'description': "Confirmed completion"
        }
    ]
    
    print("=" * 80)
    print("Pitt Corpus-Based Reminder Response Analysis")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Response: \"{test['response']}\"")
        print("-" * 80)
        
        result = analyzer.analyze_reminder_response(test['response'])
        
        print(f"   Cognitive Risk: {result['cognitive_risk_score']:.2f}")
        print(f"   Confusion: {'YES' if result['confusion_detected'] else 'NO'}")
        print(f"   Memory Issue: {'YES' if result['memory_issue_detected'] else 'NO'}")
        print(f"   Uncertainty: {'YES' if result['uncertainty_detected'] else 'NO'}")
        print(f"   Interaction Type: {result['interaction_type']}")
        print(f"   Recommended Action: {result['recommended_action']}")
        print(f"   Caregiver Alert: {'YES' if result['caregiver_alert_needed'] else 'NO'}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_reminder_analyzer()
