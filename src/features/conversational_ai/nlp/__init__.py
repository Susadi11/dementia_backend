"""
NLP Module for Conversational AI Analysis

Provides comprehensive text processing and analysis for dementia detection:
- Text preprocessing (normalization, tokenization, lemmatization)
- Semantic analysis with BERT embeddings
- Sentiment and emotion detection
- Discourse and linguistic feature extraction
- Dementia marker identification

Main entry point: NLPEngine

Example:
    from src.features.conversational_ai.nlp import NLPEngine

    engine = NLPEngine()
    result = engine.analyze("Hello, I'm feeling good today")
    markers = engine.extract_dementia_markers(result)
"""

from .nlp_engine import NLPEngine, NLPAnalysisResult
from .utils.text_preprocessing import TextPreprocessor, PreprocessedText
from .processors.semantic_analyzer import SemanticAnalyzer, SemanticAnalysis
from .processors.sentiment_analyzer import SentimentAnalyzer, EmotionScores
from .processors.syntax_analyzer import LinguisticAnalyzer, LinguisticFeatures

__all__ = [
    'NLPEngine',
    'NLPAnalysisResult',
    'TextPreprocessor',
    'PreprocessedText',
    'SemanticAnalyzer',
    'SemanticAnalysis',
    'SentimentAnalyzer',
    'EmotionScores',
    'LinguisticAnalyzer',
    'LinguisticFeatures',
]
