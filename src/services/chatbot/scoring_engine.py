from typing import Dict, Any, Optional, List
from datetime import datetime
import re
import logging

from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Optional semantic similarity for P1 and P12
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False
    logger.warning("sentence-transformers not available - using basic word overlap for P1 and P12")

# Optional TextBlob sentiment for P7
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available - P7 will use basic emotion word detection")


class ScoringEngine:

    def __init__(self):
        # Initialize audio feature extractor
        self.audio_processor = AudioProcessor()

        # P1: filler word marker list
        self.incoherence_markers = [
            "anyway", "whatever", "you know", "like", "um",
            "basically", "literally", "actually", "uh", "er"
        ]

        # P2: question word detection list
        self.question_words = ["what", "when", "where", "who", "why", "how"]

        # P3: self-correction phrase patterns
        self.correction_patterns = [
            r"i mean", r"no wait", r"actually", r"sorry i meant",
            r"let me rephrase", r"correction", r"i said .+ but",
            r"that's not right", r"no that's wrong", r"that's incorrect",
            r"not exactly", r"not quite", r"that's not what i meant",
            r"oops", r"my mistake", r"i misspoke", r"my bad",
            r"i was wrong", r"i got that wrong",
            r"rather", r"or should i say", r"let me correct",
            r"to be more accurate", r"more precisely", r"to clarify",
            r"scratch that", r"forget (what )?i (just )?said",
            r"disregard that", r"never mind", r"ignore that",
            r"let me start over", r"let me try again"
        ]

        # P4: low-confidence language marker list
        self.uncertainty_markers = [
            "maybe", "i think", "probably", "perhaps", "might",
            "could be", "not sure", "i guess", "possibly",
            "i believe", "seems like", "appears to be",
            "looks like", "sounds like", "feels like",
            "kind of", "sort of", "somewhat", "a bit",
            "a little", "fairly", "quite", "rather",
            "i suppose", "i assume", "i would say",
            "i suspect", "i imagine", "presumably",
            "uncertain", "unclear", "doubtful", "unsure",
            "hard to say", "difficult to tell", "can't say for sure",
            "not certain", "not confident", "don't know for sure"
        ]

        # P7: emotion keyword detection lists
        self.emotion_words = [
            "angry", "mad", "furious", "irritated", "annoyed",
            "sad", "depressed", "upset", "unhappy", "miserable",
            "frustrated", "confused", "overwhelmed", "stressed",
            "anxious", "worried", "nervous", "scared", "afraid",
            "lost", "disoriented", "bewildered", "puzzled",
            "perplexed", "baffled", "mixed up",
            "sorry", "apologetic", "embarrassed", "ashamed",
            "helpless", "hopeless", "desperate"
        ]

        # P7: speech slip and error words
        self.slip_indicators = [
            "oops", "whoops", "oh no", "uh oh",
            "wrong", "mistake", "error", "my bad",
            "messed up", "screwed up", "goofed",
            "wait", "hold on", "hang on",
            "what", "huh", "um", "uh", "er",
            "what's the word", "you know that thing",
            "whatchamacallit", "thingy", "thing",
            "whatshisname", "whatsername"
        ]

        # P11: memory failure phrase patterns
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

        # Session message and question tracking
        self.conversation_history: List[str] = []
        self.question_history: List[str] = []

        # Load semantic similarity model optionally
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
        if timestamp is None:
            timestamp = datetime.now()

        # Update conversation history with context
        if conversation_context:
            self.conversation_history = conversation_context
        self.conversation_history.append(text)

        # Extract audio features from file or bytes
        extracted_audio_features = {}
        if audio_path:
            extracted_audio_features = self.audio_processor.extract_features_from_file(audio_path)
        elif audio_bytes:
            extracted_audio_features = self.audio_processor.extract_features_from_bytes(audio_bytes)

        # Merge extracted and provided features; provided takes precedence
        final_audio_features = {}
        if extracted_audio_features:
            final_audio_features.update(extracted_audio_features)
        if audio_features:
            final_audio_features.update(audio_features)

        text_lower = text.lower().strip()

        # Score all 12 behavioral parameters
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

        # Sum all parameter scores (max 36)
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
        # Score filler density and topic jump
        score = 0
        words = text.split()

        if len(words) == 0:
            return 0

        # Filler word density normalized by message length
        filler_count = sum(1 for w in words if w in self.incoherence_markers)
        filler_density = filler_count / len(words)

        if filler_density > 0.25:
            score += 1
        elif filler_density > 0.15:
            score += 0.5

        # Topic jump from previous message
        if len(self.conversation_history) >= 2:
            prev_msg = self.conversation_history[-2]

            if self.semantic_model is not None:
                try:
                    embeddings = self.semantic_model.encode([prev_msg, text])
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    if similarity < 0.3:
                        score += 2
                    elif similarity < 0.5:
                        score += 1
                except Exception as e:
                    logger.error(f"Error calculating semantic similarity: {e}")
                    score += self._word_overlap_score(prev_msg, text)
            else:
                score += self._word_overlap_score(prev_msg, text)

        return min(int(score), 3)

    def _word_overlap_score(self, prev_msg: str, curr_msg: str) -> float:
        # Topic jump score via word overlap ratio
        prev_words = set(prev_msg.lower().split())
        curr_words = set(curr_msg.lower().split())

        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it'}
        prev_words = prev_words - stopwords
        curr_words = curr_words - stopwords

        if len(prev_words) == 0 or len(curr_words) == 0:
            return 0.0

        overlap_ratio = len(prev_words.intersection(curr_words)) / max(len(prev_words), len(curr_words))

        if overlap_ratio < 0.15:
            return 2.0
        elif overlap_ratio < 0.30:
            return 1.0
        return 0.0

    def _score_repeated_questions(self, text: str) -> int:
        # Track and compare recent question history
        if "?" in text:
            self.question_history.append(text)

        if len(self.question_history) < 2:
            return 0

        recent_questions = self.question_history[-5:]
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
        # Word overlap similarity threshold 0.6
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2), 1)
        return similarity > 0.6

    def _score_self_correction(self, text: str) -> int:
        # Count matched self-correction patterns
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
        # Count uncertainty marker occurrences in text
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
        # Match explicit memory failure phrases
        memory_failure_count = sum(
            1 for pattern in self.memory_failure_patterns
            if re.search(pattern, text, re.IGNORECASE)
        )
        if memory_failure_count >= 2:
            return 3
        elif memory_failure_count >= 1:
            return 2

        # Implicit memory issue: asking about own prior speech
        if "what did i" in text or "what was i" in text:
            return 1

        return 0

    def _score_topic_maintenance(self, text: str) -> int:
        # Check topic drift across recent messages
        if len(self.conversation_history) < 3:
            return 0

        recent_messages = self.conversation_history[-3:]

        if self.semantic_model is not None:
            try:
                embeddings = self.semantic_model.encode(recent_messages)
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    similarities.append(sim)

                avg_similarity = sum(similarities) / len(similarities)

                if avg_similarity < 0.4:
                    return 3
                elif avg_similarity < 0.6:
                    return 2
                elif avg_similarity < 0.75:
                    return 1
                return 0

            except Exception as e:
                logger.error(f"Error calculating topic maintenance: {e}")
                return self._score_topic_maintenance_word_overlap(recent_messages)
        else:
            return self._score_topic_maintenance_word_overlap(recent_messages)

    def _score_topic_maintenance_word_overlap(self, messages: List[str]) -> int:
        # Word overlap without stopwords across messages
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those'
        }

        word_sets = []
        for msg in messages:
            words = set(msg.lower().split()) - stopwords
            word_sets.append(words)

        overlaps = []
        for i in range(len(word_sets) - 1):
            if len(word_sets[i]) == 0 or len(word_sets[i+1]) == 0:
                overlaps.append(0.0)
                continue
            common = word_sets[i].intersection(word_sets[i+1])
            overlap_ratio = len(common) / max(len(word_sets[i]), len(word_sets[i+1]))
            overlaps.append(overlap_ratio)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0

        if avg_overlap < 0.1:
            return 3
        elif avg_overlap < 0.2:
            return 2
        elif avg_overlap < 0.3:
            return 1
        return 0

    # ==================== AUDIO-BASED PARAMETERS ====================

    def _score_hesitation_pauses(self, audio_features: Optional[Dict[str, Any]]) -> int:
        # Map pause frequency to 0-3 score
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
        # Map tremor intensity to 0-3 score
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
        # Normal speech 120-150 wpm benchmark
        if not audio_features:
            return 0
        speech_rate = audio_features.get("speech_rate", 120)
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
        # Combine sentiment, slips, and audio score

        # Emotional intensity from text sentiment
        emotion_score = 0
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = abs(blob.sentiment.polarity)
                subjectivity = blob.sentiment.subjectivity
                if polarity > 0.5 and subjectivity > 0.6:
                    emotion_score = 2
                elif polarity > 0.3 or subjectivity > 0.5:
                    emotion_score = 1
            except Exception as e:
                logger.error(f"TextBlob sentiment analysis failed: {e}")
                emotion_score = self._emotion_keyword_score(text)
        else:
            emotion_score = self._emotion_keyword_score(text)

        # Speech slip error word count
        slip_score = 0
        slip_count = sum(1 for word in self.slip_indicators if word in text.lower())
        if slip_count >= 3:
            slip_score = 2
        elif slip_count >= 1:
            slip_score = 1

        # Audio emotion intensity signal
        audio_score = 0
        if audio_features:
            emotion_intensity = audio_features.get("emotion_intensity", 0)
            if emotion_intensity > 0.7:
                audio_score = 2
            elif emotion_intensity > 0.4:
                audio_score = 1

        combined_score = emotion_score + slip_score + audio_score

        if combined_score >= 5:
            return 3
        elif combined_score >= 3:
            return 2
        elif combined_score >= 1:
            return 1
        return 0

    def _emotion_keyword_score(self, text: str) -> int:
        # Keyword-based emotion intensity fallback
        emotion_count = sum(1 for word in self.emotion_words if word in text.lower())
        if emotion_count >= 3:
            return 2
        elif emotion_count >= 1:
            return 1
        return 0

    # ==================== TIME-BASED PARAMETERS ====================

    def _score_evening_errors(self, timestamp: datetime) -> int:
        # Sundowning effect score by time of day
        hour = timestamp.hour
        if 20 <= hour or hour < 6:
            if len(self.conversation_history) >= 3:
                return 2
            return 1
        elif 16 <= hour < 20:
            if len(self.conversation_history) >= 5:
                return 1
        return 0

    def _score_in_session_decline(self) -> int:
        # First half vs second half message comparison
        if len(self.conversation_history) < 4:
            return 0

        mid_point = len(self.conversation_history) // 2
        first_half = self.conversation_history[:mid_point]
        second_half = self.conversation_history[mid_point:]

        first_avg_len = sum(len(msg.split()) for msg in first_half) / len(first_half)
        second_avg_len = sum(len(msg.split()) for msg in second_half) / len(second_half)

        # Shorter messages in second half = decline
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
        # Map numeric scores to human-readable labels
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
        # Clear history for a new session
        self.conversation_history = []
        self.question_history = []
