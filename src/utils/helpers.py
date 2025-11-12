"""
Helper Functions and Utilities

Provides utility functions for dementia risk assessment and report generation.
"""

from typing import Dict, List, Tuple
import json


def dementia_risk_level(risk_score: float) -> Tuple[str, str]:
    """
    Determine dementia risk level from score.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        Tuple of (risk_level, description)
    """
    if risk_score < 0.3:
        return 'low', 'Low risk of dementia indicators'
    elif risk_score < 0.6:
        return 'moderate', 'Moderate risk - recommend further evaluation'
    else:
        return 'high', 'High risk indicators detected - consult healthcare provider'


def format_features_for_display(features: Dict) -> Dict:
    """
    Format features dictionary for API response.
    
    Args:
        features: Raw features dictionary
        
    Returns:
        Formatted features for display
    """
    formatted = {}
    
    feature_descriptions = {
        'semantic_incoherence': 'Lack of logical coherence in speech',
        'repeated_questions': 'Asking the same questions repeatedly',
        'self_correction': 'Frequency of self-corrections',
        'low_confidence_answer': 'Uncertainty indicators in responses',
        'hesitation_pauses': 'Number of hesitation pauses',
        'vocal_tremors': 'Tremors detected in voice',
        'emotion_slip': 'Emotional intensity combined with speech errors',
        'slowed_speech': 'Reduced speech rate',
        'evening_errors': 'Errors increasing during evening hours',
        'in_session_decline': 'Deterioration within conversation session'
    }
    
    for feature, value in features.items():
        formatted[feature] = {
            'value': round(value, 3),
            'percentage': round(value * 100, 1),
            'description': feature_descriptions.get(feature, feature)
        }
    
    return formatted


def calculate_overall_risk(features: Dict) -> float:
    """
    Calculate overall dementia risk from features.
    
    Args:
        features: Extracted features dictionary
        
    Returns:
        Overall risk score (0-1)
    """
    if not features:
        return 0.0
    
    # Weight the features
    weights = {
        'semantic_incoherence': 0.12,
        'repeated_questions': 0.12,
        'self_correction': 0.10,
        'low_confidence_answer': 0.10,
        'hesitation_pauses': 0.10,
        'vocal_tremors': 0.10,
        'emotion_slip': 0.10,
        'slowed_speech': 0.10,
        'evening_errors': 0.08,
        'in_session_decline': 0.08
    }
    
    risk_score = 0.0
    for feature, weight in weights.items():
        if feature in features:
            risk_score += features[feature] * weight
    
    return min(risk_score, 1.0)


def generate_report(features: Dict, risk_score: float) -> Dict:
    """
    Generate comprehensive analysis report.
    
    Args:
        features: Extracted features
        risk_score: Overall risk score
        
    Returns:
        Report dictionary
    """
    risk_level, risk_description = dementia_risk_level(risk_score)

    high_indicators = [
        (feature, value) for feature, value in features.items()
        if value > 0.6
    ]
    
    report = {
        'overall_risk_score': round(risk_score, 3),
        'risk_percentage': round(risk_score * 100, 1),
        'risk_level': risk_level,
        'risk_description': risk_description,
        'key_indicators': [
            {
                'indicator': indicator,
                'score': round(value, 3),
                'percentage': round(value * 100, 1)
            }
            for indicator, value in high_indicators
        ],
        'recommendations': _get_recommendations(risk_level, high_indicators)
    }
    
    return report


def _get_recommendations(risk_level: str, high_indicators: List) -> List[str]:
    """Get recommendations based on risk level."""
    recommendations = []
    
    if risk_level == 'low':
        recommendations.append('Continue regular health monitoring')
        recommendations.append('Maintain cognitive activities')
    
    elif risk_level == 'moderate':
        recommendations.append('Schedule cognitive assessment with healthcare provider')
        recommendations.append('Monitor speech and language patterns')
        recommendations.append('Consider neuropsychological testing')
    
    else:  # high risk
        recommendations.append('Consult with neurologist or geriatrician immediately')
        recommendations.append('Conduct comprehensive neuropsychological evaluation')
        recommendations.append('Consider MRI or other diagnostic imaging')
        recommendations.append('Start appropriate medical interventions if needed')
    
    return recommendations


__all__ = [
    "dementia_risk_level",
    "format_features_for_display",
    "calculate_overall_risk",
    "generate_report"
]
