"""
NLP Engine - Main Orchestrator

Coordinates all NLP analysis components:
- Text preprocessing
- Semantic analysis
- Sentiment/emotion analysis
- Discourse and linguistic analysis

Provides a unified interface for comprehensive text processing.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import warnings

from .utils.text_preprocessing import TextPreprocessor, PreprocessedText
from .processors.semantic_analyzer import SemanticAnalyzer, SemanticAnalysis
from .processors.sentiment_analyzer import SentimentAnalyzer, EmotionScores
from .processors.syntax_analyzer import LinguisticAnalyzer, LinguisticFeatures

logger = logging.getLogger(__name__)


@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis result"""
    original_text: str
    preprocessed_text: PreprocessedText
    semantic_analysis: SemanticAnalysis
    emotion_analysis: EmotionScores
    linguistic_features: LinguisticFeatures

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete analysis to dictionary"""
        return {
            'original_text': self.original_text,
            'preprocessed': self.preprocessed_text.to_dict(),
            'semantic': self.semantic_analysis.to_dict(),
            'emotion': self.emotion_analysis.to_dict(),
            'linguistic': self.linguistic_features.to_dict(),
        }


class NLPEngine:
    """
    Main NLP Engine that orchestrates all text analysis components.

    Features:
    - Text normalization and preprocessing
    - Semantic analysis with BERT embeddings
    - Sentiment and emotion detection
    - Discourse and linguistic feature extraction
    - Dementia marker detection

    Usage:
        engine = NLPEngine()
        result = engine.analyze(text)
        features = engine.extract_dementia_markers(result)
    """

    def __init__(self, spacy_model: str = "en_core_web_sm",
                 semantic_model: str = "distilbert-base-uncased",
                 enable_semantic: bool = True,
                 enable_emotion: bool = True,
                 enable_linguistic: bool = True,
                 device: str = "cpu"):
        """
        Initialize NLPEngine.

        Args:
            spacy_model: SpaCy model name
            semantic_model: Semantic model name
            enable_semantic: Enable semantic analysis
            enable_emotion: Enable emotion analysis
            enable_linguistic: Enable linguistic analysis
            device: Device to use ('cpu' or 'cuda')
        """
        self.spacy_model = spacy_model
        self.semantic_model = semantic_model
        self.device = device

        # Initialize components
        logger.info("Initializing NLPEngine components...")

        self.text_processor = TextPreprocessor(spacy_model=spacy_model)
        logger.info("[OK] TextPreprocessor initialized")

        if enable_semantic:
            self.semantic_analyzer = SemanticAnalyzer(
                model_name=semantic_model,
                device=device
            )
            logger.info("[OK] SemanticAnalyzer initialized")
        else:
            self.semantic_analyzer = None

        if enable_emotion:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("[OK] SentimentAnalyzer initialized")
        else:
            self.sentiment_analyzer = None

        if enable_linguistic:
            nlp_obj = self.text_processor.nlp if self.text_processor else None
            self.linguistic_analyzer = LinguisticAnalyzer(nlp_model=nlp_obj)
            logger.info("[OK] LinguisticAnalyzer initialized")
        else:
            self.linguistic_analyzer = None

        logger.info("NLPEngine initialization complete")

    def analyze(self, text: str, include_embeddings: bool = False) -> NLPAnalysisResult:
        """
        Perform complete NLP analysis on text.

        Args:
            text: Text to analyze
            include_embeddings: Include embedding vectors in output

        Returns:
            NLPAnalysisResult with all analyses
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Step 1: Preprocessing
        logger.debug(f"Preprocessing text ({len(text)} chars)")
        preprocessed = self.text_processor.process(text)

        # Step 2: Semantic analysis
        semantic_analysis = None
        if self.semantic_analyzer:
            try:
                logger.debug("Performing semantic analysis")
                semantic_analysis = self.semantic_analyzer.calculate_semantic_coherence(
                    sentences=preprocessed.sentences
                )
                # Clear embeddings if not needed
                if not include_embeddings:
                    semantic_analysis.embeddings = None
            except Exception as e:
                logger.error(f"Semantic analysis error: {e}")
                semantic_analysis = self.semantic_analyzer._empty_semantic_analysis()

        # Step 3: Emotion and sentiment analysis
        emotion_analysis = None
        if self.sentiment_analyzer:
            try:
                logger.debug("Performing emotion analysis")
                emotion_analysis = self.sentiment_analyzer.analyze(
                    text=text,
                    sentences=preprocessed.sentences,
                    tokens=preprocessed.tokens
                )
            except Exception as e:
                logger.error(f"Emotion analysis error: {e}")
                emotion_analysis = self.sentiment_analyzer._empty_emotion_scores()

        # Step 4: Linguistic analysis
        linguistic_features = None
        if self.linguistic_analyzer:
            try:
                logger.debug("Performing linguistic analysis")
                linguistic_features = self.linguistic_analyzer.analyze(
                    text=text,
                    sentences=preprocessed.sentences,
                    tokens=preprocessed.tokens,
                    pos_tags=preprocessed.pos_tags
                )
            except Exception as e:
                logger.error(f"Linguistic analysis error: {e}")
                linguistic_features = None

        return NLPAnalysisResult(
            original_text=text,
            preprocessed_text=preprocessed,
            semantic_analysis=semantic_analysis,
            emotion_analysis=emotion_analysis,
            linguistic_features=linguistic_features,
        )

    def extract_dementia_markers(self, analysis: NLPAnalysisResult) -> Dict[str, float]:
        """
        Extract dementia-relevant markers from NLP analysis.

        Dementia-related features:
        1. semantic_incoherence - Difficulty maintaining coherent topics
        2. semantic_drift - Topic shifting over time
        3. repetition_score - Repeated words/phrases
        4. hesitation_score - Filled pauses and hesitations
        5. low_confidence_score - Uncertain language patterns
        6. emotional_drift - Changes in emotional tone
        7. discourse_complexity - Reduced ability to form complex structures
        8. word_finding_difficulty - Lower lexical diversity
        9. tangentiality - Topic-related incoherence
        10. circumlocution - Using many words to express simple ideas

        Args:
            analysis: NLPAnalysisResult from analyze()

        Returns:
            Dict of marker_name -> score (0-1)
        """
        markers = {}

        # 1. Semantic Incoherence
        if analysis.semantic_analysis:
            markers['semantic_incoherence'] = analysis.semantic_analysis.incoherence_score
        else:
            markers['semantic_incoherence'] = 0.0

        # 2. Semantic Drift
        if analysis.semantic_analysis:
            markers['semantic_drift'] = analysis.semantic_analysis.semantic_drift
        else:
            markers['semantic_drift'] = 0.0

        # 3. Repetition Score
        if analysis.linguistic_features:
            # Normalize repetition count to 0-1
            max_reps = len(analysis.preprocessed_text.tokens)
            markers['repetition_score'] = min(1.0, analysis.linguistic_features.repetition_count / max(1, max_reps))
        else:
            markers['repetition_score'] = 0.0

        # 4. Hesitation Score
        if analysis.emotion_analysis:
            markers['hesitation_score'] = analysis.emotion_analysis.hesitation_score
        else:
            markers['hesitation_score'] = 0.0

        # 5. Low Confidence Score
        if analysis.emotion_analysis:
            # Inverse of confidence
            markers['low_confidence_score'] = 1.0 - analysis.emotion_analysis.confidence_level
        else:
            markers['low_confidence_score'] = 0.0

        # 6. Emotional Drift
        if analysis.emotion_analysis:
            # Number of emotion shifts / number of sentences
            num_shifts = len(analysis.emotion_analysis.emotion_shifts)
            num_sentences = len(analysis.preprocessed_text.sentences)
            markers['emotional_drift'] = min(1.0, num_shifts / max(1, num_sentences))
        else:
            markers['emotional_drift'] = 0.0

        # 7. Discourse Complexity (inverse - lower complexity = lower score)
        if analysis.linguistic_features:
            # Lower complexity and lower TTR indicate reduced language ability
            complexity = analysis.linguistic_features.syntax_complexity
            ttr = analysis.linguistic_features.type_token_ratio
            markers['reduced_complexity'] = (1.0 - complexity + 1.0 - ttr) / 2
        else:
            markers['reduced_complexity'] = 0.0

        # 8. Word Finding Difficulty (low lexical diversity)
        if analysis.linguistic_features:
            # Low TTR = poor word finding ability
            markers['word_finding_difficulty'] = 1.0 - analysis.linguistic_features.type_token_ratio
        else:
            markers['word_finding_difficulty'] = 0.0

        # 9. Tangentiality (semantic incoherence related to topics)
        if analysis.semantic_analysis:
            # Use low coherence regions as indicator
            num_low_coherence = len(analysis.semantic_analysis.low_coherence_regions)
            num_sentences = len(analysis.preprocessed_text.sentences)
            markers['tangentiality'] = min(1.0, num_low_coherence / max(1, num_sentences / 2))
        else:
            markers['tangentiality'] = 0.0

        # 10. Circumlocution (using many words for simple ideas)
        # Indicated by high avg sentence length + low semantic coherence
        if analysis.linguistic_features and analysis.semantic_analysis:
            normalized_sent_length = min(1.0, analysis.linguistic_features.avg_sentence_length / 20)
            incoherence = analysis.semantic_analysis.incoherence_score
            markers['circumlocution'] = (normalized_sent_length + incoherence) / 2
        else:
            markers['circumlocution'] = 0.0

        return markers

    def extract_speech_quality_metrics(self, analysis: NLPAnalysisResult) -> Dict[str, float]:
        """
        Extract speech quality metrics for overall assessment.

        Args:
            analysis: NLPAnalysisResult from analyze()

        Returns:
            Dict of metric_name -> score
        """
        metrics = {}

        # Coherence metrics
        if analysis.semantic_analysis:
            metrics['semantic_coherence'] = analysis.semantic_analysis.semantic_coherence
        else:
            metrics['semantic_coherence'] = 0.5

        if analysis.linguistic_features:
            metrics['discourse_coherence'] = analysis.linguistic_features.discourse_coherence
        else:
            metrics['discourse_coherence'] = 0.5

        # Language complexity
        if analysis.linguistic_features:
            metrics['syntax_complexity'] = analysis.linguistic_features.syntax_complexity
            metrics['lexical_diversity'] = analysis.linguistic_features.type_token_ratio
            metrics['avg_sentence_length'] = min(1.0, analysis.linguistic_features.avg_sentence_length / 25)
        else:
            metrics['syntax_complexity'] = 0.5
            metrics['lexical_diversity'] = 0.5
            metrics['avg_sentence_length'] = 0.5

        # Emotional metrics
        if analysis.emotion_analysis:
            metrics['emotional_expressiveness'] = analysis.emotion_analysis.emotional_expressiveness
            metrics['emotional_stability'] = 1.0 - (len(analysis.emotion_analysis.emotion_shifts) /
                                                     max(1, len(analysis.preprocessed_text.sentences)))
            metrics['confidence'] = analysis.emotion_analysis.confidence_level
        else:
            metrics['emotional_expressiveness'] = 0.5
            metrics['emotional_stability'] = 0.5
            metrics['confidence'] = 0.5

        return metrics

    def get_feature_summary(self, analysis: NLPAnalysisResult) -> Dict[str, Any]:
        """
        Get a human-readable summary of all extracted features.

        Args:
            analysis: NLPAnalysisResult from analyze()

        Returns:
            Dict with organized feature summaries
        """
        summary = {
            'text_statistics': {
                'length': len(analysis.original_text),
                'word_count': analysis.preprocessed_text.word_count,
                'unique_words': analysis.preprocessed_text.unique_words,
                'sentence_count': len(analysis.preprocessed_text.sentences),
                'language': analysis.preprocessed_text.language,
            },
            'semantic_features': analysis.semantic_analysis.to_dict() if analysis.semantic_analysis else {},
            'emotion_features': analysis.emotion_analysis.to_dict() if analysis.emotion_analysis else {},
            'linguistic_features': analysis.linguistic_features.to_dict() if analysis.linguistic_features else {},
            'dementia_markers': self.extract_dementia_markers(analysis),
            'speech_quality': self.extract_speech_quality_metrics(analysis),
        }

        return summary

    def batch_analyze(self, texts: List[str],
                     include_embeddings: bool = False) -> List[NLPAnalysisResult]:
        """
        Analyze multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            include_embeddings: Include embeddings in results

        Returns:
            List of NLPAnalysisResult objects
        """
        results = []
        for i, text in enumerate(texts):
            try:
                logger.debug(f"Analyzing text {i+1}/{len(texts)}")
                result = self.analyze(text, include_embeddings=include_embeddings)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text {i+1}: {e}")
                # Return empty result on error
                results.append(None)

        return [r for r in results if r is not None]
