"""
Scoring Engine for 12 Behavioral Parameters

Analyzes user text and audio input to score 12 dementia indicators.
Each parameter scored 0-3 (0=none, 1=mild, 2=moderate, 3=severe).
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import re
import logging
from collections import Counter

# Import AudioProcessor
from src.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False
    logger.warning("sentence-transformers not available - using basic word overlap for P1 and P12")

# Try to import TextBlob for sentiment analysis (P7)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available - P7 will use basic emotion word detection")


class ScoringEngine:
    """
    Scores 12 behavioral parameters from text and audio analysis.

    Parameters (0-3 scale):
    1. Semantic incoherence (text)
    2. Repeated questions (text)
    3. Self-correction (text)
    4. Low confidence answers (text)
    5. Hesitation pauses (audio)
    6. Vocal tremors (audio)
    7. Emotion + slip (text + audio)
    8. Slowed speech (audio)
    9. Evening errors (time-based)
    10. In-session decline (text pattern)
    11. Memory recall failure (text)
    12. Topic maintenance failure (text)
    """

    def __init__(self):
        """Initialize scoring engine with detection patterns"""
        
        # Initialize AudioProcessor
        self.audio_processor = AudioProcessor()

        # P1: Semantic incoherence patterns
        self.incoherence_markers = [
            "anyway", "whatever", "you know", "like", "um",
            "basically", "literally", "actually", "uh", "er"
        ]

        # P2: Question patterns for repetition detection
        self.question_words = ["what", "when", "where", "who", "why", "how"]

        # P3: Self-correction patterns (ENHANCED - 25 patterns)
        self.correction_patterns = [
            # Direct corrections
            r"i mean", r"no wait", r"actually", r"sorry i meant",
            r"let me rephrase", r"correction", r"i said .+ but",
            
            # Negation of previous statement
            r"that's not right", r"no that's wrong", r"that's incorrect",
            r"not exactly", r"not quite", r"that's not what i meant",
            
            # Mistake acknowledgment
            r"oops", r"my mistake", r"i misspoke", r"my bad",
            r"i was wrong", r"i got that wrong",
            
            # Rephrasing indicators
            r"rather", r"or should i say", r"let me correct",
            r"to be more accurate", r"more precisely", r"to clarify",
            
            # Retractions
            r"scratch that", r"forget (what )?i (just )?said", 
            r"disregard that", r"never mind", r"ignore that",
            r"let me start over", r"let me try again"
        ]

        # P4: Low confidence markers (ENHANCED - 35 markers)
        self.uncertainty_markers = [
            # Basic uncertainty
            "maybe", "i think", "probably", "perhaps", "might",
            "could be", "not sure", "i guess", "possibly",
            
            # Belief/appearance
            "i believe", "seems like", "appears to be",
            "looks like", "sounds like", "feels like",
            
            # Hedging
            "kind of", "sort of", "somewhat", "a bit",
            "a little", "fairly", "quite", "rather",
            
            # Supposition
            "i suppose", "i assume", "i would say",
            "i suspect", "i imagine", "presumably",
            
            # Explicit uncertainty
            "uncertain", "unclear", "doubtful", "unsure",
            "hard to say", "difficult to tell", "can't say for sure",
            "not certain", "not confident", "don't know for sure"
        ]

        # P7: Emotion words (ENHANCED - 30+ emotions)
        self.emotion_words = [
            # Negative emotions
            "angry", "mad", "furious", "irritated", "annoyed",
            "sad", "depressed", "upset", "unhappy", "miserable",
            "frustrated", "confused", "overwhelmed", "stressed",
            "anxious", "worried", "nervous", "scared", "afraid",
            
            # Confusion/disorientation
            "lost", "disoriented", "bewildered", "puzzled",
            "perplexed", "baffled", "mixed up",
            
            # Distress
            "sorry", "apologetic", "embarrassed", "ashamed",
            "helpless", "hopeless", "desperate"
        ]
        
        # P7: Speech slip/error indicators (NEW - 20+ patterns)
        self.slip_indicators = [
            # Explicit error acknowledgment
            "oops", "whoops", "oh no", "uh oh",
            "wrong", "mistake", "error", "my bad",
            "messed up", "screwed up", "goofed",
            
            # Confusion markers
            "wait", "hold on", "hang on",
            "what", "huh", "um", "uh", "er",
            
            # Loss of words
            "what's the word", "you know that thing",
            "whatchamacallit", "thingy", "thing",
            "whatshisname", "whatsername"
        ]

        # P11: Memory indicators (EXPANDED)
        self.memory_failure_patterns = [
            r"i (don't|can't) remember", r"i forgot",
            r"what was i saying", r"where was i",
            r"can't recall", r"slipped my mind",
            r"i have no (idea|clue|memory)",
            r"drawing a blank", r"my mind (is|went) blank",
            r"lost my train of thought",
            r"can't think of", r"escapes me",
            r"remind me", r"refresh my memory"
        ]

        # Store conversation history for context analysis
        self.conversation_history: List[str] = []
        self.question_history: List[str] = []

        # Load semantic similarity model if available
        self.semantic_model = None
        if SEMANTIC_SIMILARITY_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic similarity model loaded for P1 and P12")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None

    def analyze_session(
        self,
        text: str,
        audio_features: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        conversation_context: Optional[List[str]] = None,
        audio_path: Optional[str] = None,
        audio_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Analyze a chat session and score all 12 parameters.

        Args:
            text: User's text input
            audio_features: Optional audio analysis data (pre-calculated)
            timestamp: Session timestamp for time-based scoring
            conversation_context: Previous messages for context analysis
            audio_path: Path to audio file (for real-time analysis)
            audio_bytes: Raw audio bytes (for real-time analysis)

        Returns:
            Dictionary with parameter scores and metadata
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Update conversation history
        if conversation_context:
            self.conversation_history = conversation_context
        self.conversation_history.append(text)
        
        # Process audio if provided
        extracted_audio_features = {}
        if audio_path:
            extracted_audio_features = self.audio_processor.extract_features_from_file(audio_path)
        elif audio_bytes:
            extracted_audio_features = self.audio_processor.extract_features_from_bytes(audio_bytes)
            
        # Merge provided features with extracted features (provided takes precedence)
        final_audio_features = {}
        if extracted_audio_features:
            final_audio_features.update(extracted_audio_features)
        if audio_features:
            final_audio_features.update(audio_features)

        # Clean text
        text_lower = text.lower().strip()

        # Score all 12 parameters
        scores = {
            "p1_semantic_incoherence": self._score_semantic_incoherence(text_lower),
            "p2_repeated_questions": self._score_repeated_questions(text_lower),
            "p3_self_correction": self._score_self_correction(text_lower),
            "p4_low_confidence": self._score_low_confidence(text_lower),
            "p5_hesitation_pauses": self._score_hesitation_pauses(final_audio_features),
            "p6_vocal_tremors": self._score_vocal_tremors(final_audio_features),
            "p7_emotion_slip": self._score_emotion_slip(text_lower, final_audio_features),
            "p8_slowed_speech": self._score_slowed_speech(final_audio_features),
            "p9_evening_errors": self._score_evening_errors(timestamp),
            "p10_in_session_decline": self._score_in_session_decline(),
            "p11_memory_recall_failure": self._score_memory_failure(text_lower),
            "p12_topic_maintenance": self._score_topic_maintenance(text_lower)
        }

        # Calculate session raw score
        session_raw_score = sum(scores.values())

        return {
            "scores": scores,
            "session_raw_score": session_raw_score,
            "max_possible_score": 36,
            "timestamp": timestamp,
            "analysis_details": self._generate_analysis_details(scores)
        }

    # ==================== TEXT-BASED PARAMETERS ====================

    def _score_semantic_incoherence(self, text: str) -> int:
        """
        P1: Score semantic incoherence (0-3).
        IMPROVED: Detects actual semantic jumps + filler density.

        Method:
        1. Calculate filler word density (normalized by message length)
        2. Check sentence coherence within message
        3. Check topic jump from previous message (semantic or word overlap)
        """
        score = 0
        words = text.split()

        if len(words) == 0:
            return 0

        # Component 1: Filler word density (normalized by length)
        filler_count = sum(1 for w in words if w in self.incoherence_markers)
        filler_density = filler_count / len(words)

        if filler_density > 0.25:  # 25%+ of words are fillers = severe
            score += 1
        elif filler_density > 0.15:  # 15-25% = moderate
            score += 0.5

        # Component 2: Check topic jump from previous message
        if len(self.conversation_history) >= 2:
            prev_msg = self.conversation_history[-2]

            # Use semantic similarity if available
            if self.semantic_model is not None:
                try:
                    embeddings = self.semantic_model.encode([prev_msg, text])
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                    # Low similarity = topic jump
                    if similarity < 0.3:
                        score += 2  # Severe jump
                    elif similarity < 0.5:
                        score += 1  # Moderate jump
                except Exception as e:
                    logger.error(f"Error calculating semantic similarity: {e}")
                    # Fallback to word overlap
                    score += self._word_overlap_score(prev_msg, text)
            else:
                # Use word overlap method
                score += self._word_overlap_score(prev_msg, text)

        return min(int(score), 3)

    def _word_overlap_score(self, prev_msg: str, curr_msg: str) -> float:
        """Helper: Calculate topic jump score using word overlap"""
        prev_words = set(prev_msg.lower().split())
        curr_words = set(curr_msg.lower().split())

        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it'}
        prev_words = prev_words - stopwords
        curr_words = curr_words - stopwords

        if len(prev_words) == 0 or len(curr_words) == 0:
            return 0.0

        overlap_ratio = len(prev_words.intersection(curr_words)) / max(len(prev_words), len(curr_words))

        # Low overlap = topic jump
        if overlap_ratio < 0.15:
            return 2.0
        elif overlap_ratio < 0.30:
            return 1.0
        return 0.0

    def _score_repeated_questions(self, text: str) -> int:
        """
        P2: Score repeated questions (0-3).
        Detects asking the same question multiple times.
        """
        # Extract questions
        if "?" in text:
            self.question_history.append(text)

        # Check for repetition in recent history
        if len(self.question_history) < 2:
            return 0

        # Count similar questions in recent history
        recent_questions = self.question_history[-5:]  # Last 5 questions
        repetition_count = 0

        for i, q1 in enumerate(recent_questions):
            for q2 in recent_questions[i+1:]:
                if self._questions_similar(q1, q2):
                    repetition_count += 1

        if repetition_count >= 3:
            return 3
        elif repetition_count >= 2:
            return 2
        elif repetition_count >= 1:
            return 1
        return 0

    def _questions_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar"""
        # Simple similarity: check if they share question words and key terms
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2), 1)

        return similarity > 0.6

    def _score_self_correction(self, text: str) -> int:
        """
        P3: Score self-correction frequency (0-3).
        Detects phrases indicating correction of previous statements.
        """
        correction_count = sum(
            1 for pattern in self.correction_patterns
            if re.search(pattern, text, re.IGNORECASE)
        )

        if correction_count >= 3:
            return 3
        elif correction_count >= 2:
            return 2
        elif correction_count >= 1:
            return 1
        return 0

    def _score_low_confidence(self, text: str) -> int:
        """
        P4: Score low confidence answers (0-3).
        Detects uncertainty markers in responses.
        """
        uncertainty_count = sum(
            1 for marker in self.uncertainty_markers if marker in text
        )

        if uncertainty_count >= 4:
            return 3
        elif uncertainty_count >= 2:
            return 2
        elif uncertainty_count >= 1:
            return 1
        return 0

    def _score_memory_failure(self, text: str) -> int:
        """
        P11: Score memory recall failure (0-3).
        Detects explicit mentions of forgetting or memory issues.
        """
        memory_failure_count = sum(
            1 for pattern in self.memory_failure_patterns
            if re.search(pattern, text, re.IGNORECASE)
        )

        if memory_failure_count >= 2:
            return 3
        elif memory_failure_count >= 1:
            return 2

        # Check for implicit memory issues (asking "what did I say")
        if "what did i" in text or "what was i" in text:
            return 1

        return 0

    def _score_topic_maintenance(self, text: str) -> int:
        """
        P12: Score topic maintenance failure (0-3).
        IMPROVED: Uses semantic similarity to detect topic drift.

        Method:
        1. If semantic model available: use embeddings + cosine similarity
        2. Otherwise: use improved word overlap with stopword removal
        """
        if len(self.conversation_history) < 3:
            return 0

        recent_messages = self.conversation_history[-3:]

        # Use semantic similarity if available
        if self.semantic_model is not None:
            try:
                # Get embeddings for all recent messages
                embeddings = self.semantic_model.encode(recent_messages)

                # Calculate pairwise similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    similarities.append(sim)

                avg_similarity = sum(similarities) / len(similarities)

                # Low similarity = topic drift
                if avg_similarity < 0.4:
                    return 3
                elif avg_similarity < 0.6:
                    return 2
                elif avg_similarity < 0.75:
                    return 1
                return 0

            except Exception as e:
                logger.error(f"Error calculating topic maintenance: {e}")
                # Fallback to word overlap
                return self._score_topic_maintenance_word_overlap(recent_messages)
        else:
            # Use improved word overlap method
            return self._score_topic_maintenance_word_overlap(recent_messages)

    def _score_topic_maintenance_word_overlap(self, messages: List[str]) -> int:
        """
        Helper: Score topic maintenance using improved word overlap.
        Removes stopwords for better accuracy.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those'
        }

        # Get word sets (without stopwords)
        word_sets = []
        for msg in messages:
            words = set(msg.lower().split()) - stopwords
            word_sets.append(words)

        # Calculate overlap between consecutive messages
        overlaps = []
        for i in range(len(word_sets) - 1):
            if len(word_sets[i]) == 0 or len(word_sets[i+1]) == 0:
                overlaps.append(0.0)
                continue

            common = word_sets[i].intersection(word_sets[i+1])
            overlap_ratio = len(common) / max(len(word_sets[i]), len(word_sets[i+1]))
            overlaps.append(overlap_ratio)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0

        # Low overlap = topic jumping
        if avg_overlap < 0.1:
            return 3
        elif avg_overlap < 0.2:
            return 2
        elif avg_overlap < 0.3:
            return 1
        return 0

    # ==================== AUDIO-BASED PARAMETERS ====================

    def _score_hesitation_pauses(self, audio_features: Optional[Dict[str, Any]]) -> int:
        """
        P5: Score hesitation pauses (0-3).
        Detects pauses and filled pauses in audio.
        """
        if not audio_features:
            return 0

        pause_frequency = audio_features.get("pause_frequency", 0)

        if pause_frequency > 0.3:
            return 3
        elif pause_frequency > 0.2:
            return 2
        elif pause_frequency > 0.1:
            return 1
        return 0

    def _score_vocal_tremors(self, audio_features: Optional[Dict[str, Any]]) -> int:
        """
        P6: Score vocal tremors (0-3).
        Detects shakiness in voice from audio analysis.
        """
        if not audio_features:
            return 0

        tremor_intensity = audio_features.get("tremor_intensity", 0)

        if tremor_intensity > 0.7:
            return 3
        elif tremor_intensity > 0.5:
            return 2
        elif tremor_intensity > 0.3:
            return 1
        return 0

    def _score_slowed_speech(self, audio_features: Optional[Dict[str, Any]]) -> int:
        """
        P8: Score slowed speech (0-3).
        Normal speech: 120-150 words/min. Slower indicates issues.
        """
        if not audio_features:
            return 0

        speech_rate = audio_features.get("speech_rate", 120)  # words per minute

        if speech_rate < 80:
            return 3
        elif speech_rate < 100:
            return 2
        elif speech_rate < 110:
            return 1
        return 0

    # ==================== COMBINED PARAMETERS ====================

    def _score_emotion_slip(
        self,
        text: str,
        audio_features: Optional[Dict[str, Any]]
    ) -> int:
        """
        P7: Score emotion + slip (0-3).
        IMPROVED: Uses sentiment analysis + speech error detection.
        
        Combines:
        1. Emotional intensity (text sentiment analysis or keyword detection)
        2. Speech slips/errors (word-finding difficulties, error acknowledgments)
        3. Audio emotion intensity
        """
        score = 0
        
        # Component 1: Emotional intensity from text
        emotion_score = 0
        
        if TEXTBLOB_AVAILABLE:
            try:
                # Use TextBlob sentiment analysis
                blob = TextBlob(text)
                polarity = abs(blob.sentiment.polarity)  # -1 to 1, get absolute
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # High emotion = extreme polarity + high subjectivity
                if polarity > 0.5 and subjectivity > 0.6:
                    emotion_score = 2  # Strong emotion
                elif polarity > 0.3 or subjectivity > 0.5:
                    emotion_score = 1  # Moderate emotion
                    
            except Exception as e:
                logger.error(f"TextBlob sentiment analysis failed: {e}")
                # Fallback to keyword detection
                emotion_score = self._emotion_keyword_score(text)
        else:
            # Fallback: keyword-based emotion detection
            emotion_score = self._emotion_keyword_score(text)
        
        # Component 2: Speech slips/errors
        slip_score = 0
        
        # Count slip indicators
        slip_count = sum(1 for word in self.slip_indicators if word in text.lower())
        
        if slip_count >= 3:
            slip_score = 2  # Frequent errors
        elif slip_count >= 1:
            slip_score = 1  # Some errors
        
        # Component 3: Audio emotion intensity
        audio_score = 0
        if audio_features:
            emotion_intensity = audio_features.get("emotion_intensity", 0)
            if emotion_intensity > 0.7:
                audio_score = 2
            elif emotion_intensity > 0.4:
                audio_score = 1
        
        # Combine all components
        combined_score = emotion_score + slip_score + audio_score
        
        # Return capped score (0-3)
        if combined_score >= 5:
            return 3
        elif combined_score >= 3:
            return 2
        elif combined_score >= 1:
            return 1
        return 0
    
    def _emotion_keyword_score(self, text: str) -> int:
        """Helper: Score emotional intensity using keyword detection"""
        emotion_count = sum(1 for word in self.emotion_words if word in text.lower())
        
        if emotion_count >= 3:
            return 2  # Multiple emotional words
        elif emotion_count >= 1:
            return 1  # Some emotion
        return 0

    # ==================== TIME-BASED PARAMETERS ====================

    def _score_evening_errors(self, timestamp: datetime) -> int:
        """
        P9: Score evening errors (0-3).
        Sundowning effect: errors increase in evening/night.

        This is calculated based on time of day:
        - Morning/Afternoon: baseline (0)
        - Evening: moderate increase (1-2)
        - Night: highest increase (2-3)
        """
        hour = timestamp.hour

        if 20 <= hour or hour < 6:  # Night (8 PM - 6 AM)
            # Check conversation history for error patterns
            if len(self.conversation_history) >= 3:
                return 2  # Moderate night effect
            return 1
        elif 16 <= hour < 20:  # Evening (4 PM - 8 PM)
            if len(self.conversation_history) >= 5:
                return 1  # Mild evening effect

        return 0  # Morning/Afternoon baseline

    def _score_in_session_decline(self) -> int:
        """
        P10: Score in-session decline (0-3).
        Detects deterioration of performance within the session.
        """
        if len(self.conversation_history) < 4:
            return 0

        # Compare first half vs second half of conversation
        mid_point = len(self.conversation_history) // 2
        first_half = self.conversation_history[:mid_point]
        second_half = self.conversation_history[mid_point:]

        # Calculate average message length (proxy for coherence)
        first_avg_len = sum(len(msg.split()) for msg in first_half) / len(first_half)
        second_avg_len = sum(len(msg.split()) for msg in second_half) / len(second_half)

        # Decline = shorter messages in second half
        decline_ratio = (first_avg_len - second_avg_len) / max(first_avg_len, 1)

        if decline_ratio > 0.3:
            return 3
        elif decline_ratio > 0.2:
            return 2
        elif decline_ratio > 0.1:
            return 1
        return 0

    # ==================== HELPER METHODS ====================

    def _generate_analysis_details(self, scores: Dict[str, int]) -> Dict[str, str]:
        """Generate human-readable analysis details"""
        details = {}

        for param, score in scores.items():
            if score == 0:
                level = "None detected"
            elif score == 1:
                level = "Mild"
            elif score == 2:
                level = "Moderate"
            else:
                level = "Severe"

            details[param] = level

        return details

    def reset_session(self):
        """Reset conversation history for new session"""
        self.conversation_history = []
        self.question_history = []
