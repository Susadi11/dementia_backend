"""
Sentiment and Emotion Analysis Module

Handles:
- Sentiment analysis (positive, negative, neutral)
- Emotion detection (joy, sadness, anger, fear, etc.)
- Emotional expressiveness scoring
- Confidence and hesitation detection
- Emotional drift over time
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmotionScores:
    """Container for emotion analysis"""
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1 (negative) to +1 (positive)
    emotions: Dict[str, float]  # emotion -> confidence scores
    dominant_emotion: str
    emotional_intensity: float  # 0-1, how strong the emotion is
    confidence_level: float  # 0-1, speaker confidence
    hesitation_score: float  # 0-1, degree of hesitation
    emotional_expressiveness: float  # 0-1, how emotionally expressive
    emotion_shifts: List[Tuple[int, str, str]]  # (position, from_emotion, to_emotion)
    low_confidence_phrases: List[str]  # Phrases indicating low confidence

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'sentiment_label': self.sentiment_label,
            'sentiment_score': float(self.sentiment_score),
            'emotions': {k: float(v) for k, v in self.emotions.items()},
            'dominant_emotion': self.dominant_emotion,
            'emotional_intensity': float(self.emotional_intensity),
            'confidence_level': float(self.confidence_level),
            'hesitation_score': float(self.hesitation_score),
            'emotional_expressiveness': float(self.emotional_expressiveness),
            'emotion_shifts': self.emotion_shifts,
            'low_confidence_phrases': self.low_confidence_phrases,
        }


class SentimentAnalyzer:
    """
    Sentiment and emotion analysis for detecting emotional states and confidence levels.
    """

    # Emotion keywords for pattern matching
    EMOTION_KEYWORDS = {
        'joy': ['happy', 'glad', 'delighted', 'pleased', 'wonderful', 'great', 'love'],
        'sadness': ['sad', 'unhappy', 'miserable', 'depressed', 'down', 'terrible'],
        'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated'],
        'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'frightened'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
        'disgust': ['disgusted', 'repulsed', 'gross', 'nasty', 'ugh'],
    }

    # Low confidence markers
    LOW_CONFIDENCE_MARKERS = {
        'maybe', 'perhaps', 'might', 'could be', 'i think', 'i guess',
        'possibly', 'sort of', 'kind of', 'not sure', 'uncertain',
        'probably', 'seems like', 'i believe', 'supposedly',
        'uh', 'um', 'er', 'hmm', 'well', 'you know', 'like'
    }

    # Hesitation markers
    HESITATION_MARKERS = {
        'um', 'uh', 'er', 'erm', 'ah', 'eh', 'hmm', 'mm', 'hmph',
        'like', 'basically', 'literally', 'you know', 'i mean'
    }

    def __init__(self, use_transformers: bool = True):
        """
        Initialize SentimentAnalyzer.

        Args:
            use_transformers: Use transformer models if available
        """
        self.use_transformers = use_transformers
        self.sentiment_pipeline = None
        self.emotion_pipeline = None

        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            self._load_transformers()

    def _load_transformers(self):
        """Load transformer-based sentiment and emotion models."""
        try:
            # Load sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Loaded sentiment analysis pipeline")
        except Exception as e:
            logger.warning(f"Could not load sentiment pipeline: {e}")
            self.sentiment_pipeline = None

        try:
            # Load emotion pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            logger.info("Loaded emotion analysis pipeline")
        except Exception as e:
            logger.warning(f"Could not load emotion pipeline: {e}")
            self.emotion_pipeline = None

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (label, score) where score is -1 to +1
        """
        if not text.strip():
            return 'neutral', 0.0

        # Try transformer-based analysis first
        if self.sentiment_pipeline is not None:
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result['label'].lower()
                score = result['score']

                # Normalize score to -1 to +1
                if label == 'negative':
                    return 'negative', -score
                else:
                    return 'positive', score
            except Exception as e:
                logger.warning(f"Sentiment pipeline error: {e}")

        # Fallback: TextBlob
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to +1
                if polarity > 0.1:
                    return 'positive', polarity
                elif polarity < -0.1:
                    return 'negative', polarity
                else:
                    return 'neutral', 0.0
            except Exception as e:
                logger.warning(f"TextBlob sentiment error: {e}")

        # Fallback: keyword matching
        return self._sentiment_from_keywords(text)

    def _sentiment_from_keywords(self, text: str) -> Tuple[str, float]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        positive_keywords = ['good', 'great', 'happy', 'wonderful', 'excellent',
                           'love', 'amazing', 'fantastic', 'better']
        negative_keywords = ['bad', 'terrible', 'sad', 'awful', 'hate',
                           'horrible', 'worse', 'disappointing', 'worse']

        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)

        if pos_count > neg_count:
            return 'positive', 0.5
        elif neg_count > pos_count:
            return 'negative', -0.5
        else:
            return 'neutral', 0.0

    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text.

        Args:
            text: Text to analyze

        Returns:
            Dict of emotion -> confidence score
        """
        if not text.strip():
            return {}

        # Try transformer-based analysis first
        if self.emotion_pipeline is not None:
            try:
                result = self.emotion_pipeline(text[:512])
                # Map to standard emotions
                emotion_map = {
                    'sadness': 'sadness',
                    'joy': 'joy',
                    'love': 'joy',
                    'anger': 'anger',
                    'fear': 'fear',
                    'surprise': 'surprise',
                    'disgust': 'disgust',
                    'neutral': 'neutral',
                }
                emotion_scores = {}
                for result_item in (result if isinstance(result, list) else [result]):
                    label = result_item['label'].lower()
                    score = result_item['score']
                    mapped_emotion = emotion_map.get(label, label)
                    emotion_scores[mapped_emotion] = score

                return emotion_scores
            except Exception as e:
                logger.warning(f"Emotion pipeline error: {e}")

        # Fallback: keyword matching
        return self._emotions_from_keywords(text)

    def _emotions_from_keywords(self, text: str) -> Dict[str, float]:
        """Keyword-based emotion detection."""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                emotion_scores[emotion] = min(1.0, count * 0.3)

        return emotion_scores

    def detect_confidence_level(self, text: str, tokens: Optional[List[str]] = None) -> float:
        """
        Detect speaker confidence level.

        Args:
            text: Text to analyze
            tokens: Pre-tokenized text (optional)

        Returns:
            Confidence score (0-1, where 1 is most confident)
        """
        text_lower = text.lower()

        # Count low-confidence markers
        low_conf_count = 0
        total_words = len(tokens) if tokens else len(text.split())

        for marker in self.LOW_CONFIDENCE_MARKERS:
            if marker in text_lower:
                low_conf_count += text_lower.count(marker)

        if total_words == 0:
            return 1.0

        # Higher marker count = lower confidence
        confidence = max(0.0, 1.0 - (low_conf_count / max(1, total_words / 2)))
        return float(confidence)

    def detect_hesitations(self, tokens: List[str]) -> Tuple[float, List[str]]:
        """
        Detect hesitation markers in speech.

        Args:
            tokens: List of tokens

        Returns:
            Tuple of (hesitation_score, hesitation_markers_found)
        """
        hesitations_found = []

        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.HESITATION_MARKERS:
                hesitations_found.append(token)

        # Calculate hesitation score
        total_tokens = len(tokens)
        hesitation_score = len(hesitations_found) / max(1, total_tokens) if total_tokens > 0 else 0.0
        hesitation_score = float(min(1.0, hesitation_score))

        return hesitation_score, hesitations_found

    def detect_emotion_shifts(self, sentences: List[str]) -> List[Tuple[int, str, str]]:
        """
        Detect shifts in emotional tone across sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of (sentence_index, prev_emotion, new_emotion) tuples
        """
        emotions = []
        for sent in sentences:
            emotion_dict = self.detect_emotions(sent)
            if emotion_dict:
                dominant = max(emotion_dict, key=emotion_dict.get)
                emotions.append(dominant)
            else:
                emotions.append('neutral')

        shifts = []
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                shifts.append((i, emotions[i-1], emotions[i]))

        return shifts

    def calculate_emotional_expressiveness(self,
                                          text: str,
                                          tokens: Optional[List[str]] = None) -> float:
        """
        Calculate how emotionally expressive the text is.

        Args:
            text: Text to analyze
            tokens: Pre-tokenized text (optional)

        Returns:
            Expressiveness score (0-1)
        """
        emotion_dict = self.detect_emotions(text)
        emotion_intensity = sum(emotion_dict.values()) if emotion_dict else 0.0

        # Count exclamations and intensifiers
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_words = len([w for w in (tokens or text.split()) if w.isupper() and len(w) > 1])

        # Count intensifiers
        intensifiers = ['very', 'really', 'so', 'extremely', 'incredibly']
        intensifier_count = sum(1 for word in intensifiers if word in text.lower())

        # Calculate expressiveness
        expressiveness = min(1.0, emotion_intensity * 0.5 +
                           (exclamation_count * 0.2) +
                           (question_count * 0.1) +
                           (caps_words * 0.1) +
                           (intensifier_count * 0.1))

        return float(expressiveness)

    def find_low_confidence_phrases(self, text: str) -> List[str]:
        """
        Find phrases indicating low confidence.

        Args:
            text: Text to analyze

        Returns:
            List of low-confidence phrases found
        """
        phrases = []
        text_lower = text.lower()

        for marker in sorted(self.LOW_CONFIDENCE_MARKERS, key=len, reverse=True):
            if marker in text_lower:
                # Find context around marker
                idx = text_lower.find(marker)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(marker) + 20)
                context = text[start:end].strip()
                if context:
                    phrases.append(context)

        return list(set(phrases))  # Remove duplicates

    def analyze(self, text: str, sentences: Optional[List[str]] = None,
               tokens: Optional[List[str]] = None) -> EmotionScores:
        """
        Complete emotion and sentiment analysis.

        Args:
            text: Full text
            sentences: List of sentences (optional)
            tokens: List of tokens (optional)

        Returns:
            EmotionScores object
        """
        if not text.strip():
            return self._empty_emotion_scores()

        # Analyze sentiment
        sentiment_label, sentiment_score = self.analyze_sentiment(text)

        # Detect emotions
        emotions = self.detect_emotions(text)
        dominant_emotion = max(emotions, key=emotions.get) if emotions else 'neutral'
        emotional_intensity = sum(emotions.values()) / len(emotions) if emotions else 0.0

        # Detect confidence
        confidence_level = self.detect_confidence_level(text, tokens)

        # Detect hesitations
        hesitation_score, _ = self.detect_hesitations(tokens or text.split())

        # Detect emotional expressiveness
        emotional_expressiveness = self.calculate_emotional_expressiveness(text, tokens)

        # Detect emotion shifts
        emotion_shifts = self.detect_emotion_shifts(sentences or [text])

        # Find low confidence phrases
        low_confidence_phrases = self.find_low_confidence_phrases(text)

        return EmotionScores(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            emotional_intensity=float(emotional_intensity),
            confidence_level=float(confidence_level),
            hesitation_score=float(hesitation_score),
            emotional_expressiveness=float(emotional_expressiveness),
            emotion_shifts=emotion_shifts,
            low_confidence_phrases=low_confidence_phrases,
        )

    def _empty_emotion_scores(self) -> EmotionScores:
        """Return empty emotion scores."""
        return EmotionScores(
            sentiment_label='neutral',
            sentiment_score=0.0,
            emotions={},
            dominant_emotion='neutral',
            emotional_intensity=0.0,
            confidence_level=0.5,
            hesitation_score=0.0,
            emotional_expressiveness=0.0,
            emotion_shifts=[],
            low_confidence_phrases=[],
        )
