"""
Text Processor Component

Processes text transcripts to extract dementia-related linguistic and semantic features
including semantic incoherence, repeated questions, self-corrections, and confidence markers.
"""

import re
from typing import Dict, List, Tuple


class TextProcessor:
    """
    Processes text transcripts to identify dementia-related
    linguistic and semantic patterns.
    """

    def __init__(self):
        """Initialize the Text Processor."""
        self.hesitation_markers = [
            "um", "uh", "er", "ah", "hmm", "erm", "err", "eh",
            "you know", "like", "i mean", "i think", "i guess"
        ]
        self.low_confidence_markers = [
            "maybe", "perhaps", "i think", "i guess", "i suppose",
            "sort of", "kind of", "something like", "not sure", "i don't know",
            "i'm not sure", "uncertain", "unsure"
        ]
        self.correction_markers = [
            "i mean", "wait", "no, ", "actually", "let me rephrase",
            "what i meant", "that is", "rather", "instead"
        ]
        self.emotional_markers = [
            "scared", "afraid", "worried", "confused", "frustrated",
            "angry", "sad", "happy", "laugh", "cry", "upset", "anxious"
        ]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'[^\w\s\?\.\!]', '', text)
        return text

    def extract_semantic_incoherence(self, text: str) -> float:
        """
        Detect semantic incoherence in speech.
        High value indicates off-topic or illogical utterances.

        Returns:
            Score 0-1, where 1 indicates high incoherence
        """
        text = self._clean_text(text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # Simple heuristic: check for very short sentences mixed with longer ones
        # and repetitive words (signs of incoherent thought)
        short_sentence_ratio = sum(1 for s in sentences if len(s.split()) < 3) / len(sentences)

        # Check for topic shifts (simple heuristic: repeated words across sentences)
        word_repetition = 0
        words = text.split()
        unique_words = set(words)
        if len(unique_words) > 0:
            word_repetition = 1 - (len(unique_words) / len(words))

        incoherence_score = (short_sentence_ratio * 0.4) + (word_repetition * 0.6)
        return min(1.0, incoherence_score)

    def extract_repeated_questions(self, text: str) -> int:
        """
        Count repeated questions in the transcript.
        Dementia patients may ask the same question multiple times.

        Returns:
            Number of repeated questions detected
        """
        text = self._clean_text(text)
        # Extract questions
        questions = re.findall(r'[^.!?]*\?', text)

        if len(questions) < 2:
            return 0

        # Find repeated questions (allowing for slight variations)
        repeated_count = 0
        for i, q1 in enumerate(questions):
            for q2 in questions[i + 1:]:
                # Simple similarity check: same words
                words1 = set(q1.split())
                words2 = set(q2.split())
                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.6:  # 60% similar
                        repeated_count += 1

        return repeated_count

    def extract_self_correction(self, text: str) -> int:
        """
        Count self-correction instances.
        Pattern: "I said X, I mean Y" or similar.

        Returns:
            Number of self-corrections detected
        """
        text = self._clean_text(text)
        correction_count = 0
        for marker in self.correction_markers:
            correction_count += len(re.findall(r'\b' + marker + r'\b', text))

        return max(0, correction_count)

    def extract_low_confidence_answers(self, text: str) -> float:
        """
        Detect low-confidence answers.
        High value indicates many hesitant/unsure responses.

        Returns:
            Score 0-1, where 1 indicates very low confidence
        """
        text = self._clean_text(text)
        sentences = re.split(r'[.!?]+', text)

        if len(sentences) == 0:
            return 0.0

        confidence_marker_count = 0
        for sentence in sentences:
            for marker in self.low_confidence_markers:
                if marker in sentence:
                    confidence_marker_count += 1

        low_confidence_score = min(1.0, confidence_marker_count / max(len(sentences), 1))
        return low_confidence_score

    def extract_hesitation_pauses(self, text: str) -> int:
        """
        Count hesitation pauses (um, uh, er, etc.).
        More pauses indicate cognitive difficulty.

        Returns:
            Number of hesitation markers detected
        """
        text = self._clean_text(text)
        pause_count = 0

        for marker in self.hesitation_markers:
            # Count occurrences of marker
            pause_count += len(re.findall(r'\b' + marker + r'\b', text))

        return pause_count

    def extract_emotion_slip(self, text: str) -> float:
        """
        Detect emotional expressions or inappropriate responses.

        Returns:
            Score 0-1, where 1 indicates high emotion/inappropriate responses
        """
        text = self._clean_text(text)
        emotion_count = 0
        words = text.split()

        for word in words:
            if word in self.emotional_markers:
                emotion_count += 1

        emotion_score = min(1.0, emotion_count / max(len(words) / 20, 1))
        return emotion_score

    def process(self, text: str) -> Dict[str, float]:
        """
        Process text to extract all linguistic features.

        Args:
            text: Input text transcript

        Returns:
            Dictionary containing extracted text features
        """
        features = {
            'semantic_incoherence': self.extract_semantic_incoherence(text),
            'repeated_questions': self.extract_repeated_questions(text),
            'self_correction': self.extract_self_correction(text),
            'low_confidence_answers': self.extract_low_confidence_answers(text),
            'hesitation_pauses': self.extract_hesitation_pauses(text),
            'emotion_slip': self.extract_emotion_slip(text),
            'evening_errors': 0.0,  # Requires time-of-day annotation
        }

        return features

    def get_feature_description(self) -> Dict[str, str]:
        """Get descriptions of extracted features."""
        return {
            'semantic_incoherence': 'Illogical or off-topic speech (0-1)',
            'repeated_questions': 'Count of repeated questions',
            'self_correction': 'Count of self-corrections',
            'low_confidence_answers': 'Proportion of hesitant responses (0-1)',
            'hesitation_pauses': 'Count of filled pauses (um, uh, er)',
            'emotion_slip': 'Inappropriate emotional expressions (0-1)',
            'evening_errors': 'Time-dependent cognitive decline indicator (requires metadata)',
        }
