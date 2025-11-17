"""
Discourse and Linguistic Analysis Module

Handles:
- Discourse markers and coherence
- Linguistic features (TTR, word frequency, syntax complexity)
- Part-of-speech analysis
- Repetition detection
- Discourse structure analysis
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import statistics

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LinguisticFeatures:
    """Container for linguistic analysis"""
    discourse_markers: List[str]  # Discourse markers found
    marker_frequency: Dict[str, int]  # Frequency of each marker
    type_token_ratio: float  # Lexical diversity (0-1)
    word_frequency: Dict[str, int]  # Word frequency distribution
    pos_distribution: Dict[str, int]  # Part-of-speech distribution
    syntax_complexity: float  # Sentence complexity score
    avg_sentence_length: float  # Average words per sentence
    avg_word_length: float  # Average characters per word
    repetition_count: int  # Number of word repetitions
    discourse_coherence: float  # Overall discourse coherence
    pronoun_usage: Dict[str, int]  # Pronoun frequency
    verb_tense_distribution: Dict[str, int]  # Past, present, future usage
    named_entity_count: int  # Count of proper nouns/entities
    clause_count: int  # Number of clauses
    passive_voice_ratio: float  # Proportion of passive voice

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'discourse_markers': self.discourse_markers,
            'marker_frequency': self.marker_frequency,
            'type_token_ratio': float(self.type_token_ratio),
            'word_frequency': self.word_frequency,
            'pos_distribution': self.pos_distribution,
            'syntax_complexity': float(self.syntax_complexity),
            'avg_sentence_length': float(self.avg_sentence_length),
            'avg_word_length': float(self.avg_word_length),
            'repetition_count': int(self.repetition_count),
            'discourse_coherence': float(self.discourse_coherence),
            'pronoun_usage': self.pronoun_usage,
            'verb_tense_distribution': self.verb_tense_distribution,
            'named_entity_count': int(self.named_entity_count),
            'clause_count': int(self.clause_count),
            'passive_voice_ratio': float(self.passive_voice_ratio),
        }


class LinguisticAnalyzer:
    """
    Analyze discourse markers and linguistic features for coherence and complexity.
    """

    # Common discourse markers
    DISCOURSE_MARKERS = {
        'temporal': ['then', 'next', 'after', 'before', 'meanwhile', 'later', 'finally'],
        'causal': ['because', 'since', 'as', 'therefore', 'thus', 'so', 'consequently'],
        'contrastive': ['but', 'however', 'although', 'while', 'whereas', 'yet', 'still'],
        'additive': ['and', 'also', 'furthermore', 'moreover', 'additionally', 'besides'],
        'exemplification': ['for example', 'for instance', 'such as', 'like', 'namely'],
        'reformulation': ['in other words', 'that is', 'i mean', 'that is to say'],
    }

    # Common pronouns
    PRONOUNS = {
        'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'],
        'second_person': ['you', 'your', 'yours'],
        'third_person': ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs'],
        'demonstrative': ['this', 'that', 'these', 'those'],
    }

    def __init__(self, nlp_model: Optional[object] = None):
        """
        Initialize LinguisticAnalyzer.

        Args:
            nlp_model: SpaCy nlp object (optional)
        """
        self.nlp = nlp_model

    def identify_discourse_markers(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Identify discourse markers in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (markers_found, frequency_dict)
        """
        text_lower = text.lower()
        markers_found = []
        marker_freq = Counter()

        for category, markers in self.DISCOURSE_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    # Count occurrences
                    count = text_lower.count(marker)
                    markers_found.extend([marker] * count)
                    marker_freq[marker] += count

        return markers_found, dict(marker_freq)

    def calculate_type_token_ratio(self, tokens: List[str]) -> float:
        """
        Calculate Type-Token Ratio (TTR) for lexical diversity.

        Args:
            tokens: List of tokens

        Returns:
            TTR score (0-1)
        """
        if len(tokens) == 0:
            return 0.0

        unique_tokens = len(set(token.lower() for token in tokens))
        total_tokens = len(tokens)

        # Normalize to 0-1 range (using log scale for fairness across text lengths)
        ttr = unique_tokens / total_tokens
        return float(min(1.0, ttr))

    def calculate_word_frequency(self, tokens: List[str]) -> Dict[str, int]:
        """
        Calculate word frequency distribution.

        Args:
            tokens: List of tokens

        Returns:
            Dict of word -> frequency
        """
        freq = Counter(token.lower() for token in tokens)
        # Return top 20 words
        return dict(freq.most_common(20))

    def analyze_syntax_complexity(self, sentences: List[str],
                                 pos_tags: Optional[List[Tuple[str, str]]] = None) -> float:
        """
        Analyze syntactic complexity of text.

        Args:
            sentences: List of sentences
            pos_tags: POS tags for tokens (optional)

        Returns:
            Complexity score (0-1)
        """
        if not sentences:
            return 0.0

        complexity_factors = []

        # Factor 1: Average sentence length
        sent_lengths = [len(sent.split()) for sent in sentences]
        if sent_lengths:
            avg_length = sum(sent_lengths) / len(sent_lengths)
            # Longer sentences = more complex (normalize)
            length_complexity = min(1.0, avg_length / 25)
            complexity_factors.append(length_complexity)

        # Factor 2: Subordination (if POS tags available)
        if pos_tags:
            subordinators = ['IN', 'WDT', 'WRB']  # SpaCy POS tags
            subordinate_count = sum(1 for _, pos in pos_tags if pos in subordinators)
            subordination_ratio = subordinate_count / max(1, len(pos_tags))
            complexity_factors.append(subordination_ratio)

        # Factor 3: Sentence variety (variation in length)
        if len(sent_lengths) > 1:
            try:
                variance = statistics.variance(sent_lengths)
                # Normalize variance
                variety = min(1.0, variance / 100)
                complexity_factors.append(variety)
            except:
                pass

        # Average all factors
        if complexity_factors:
            return float(sum(complexity_factors) / len(complexity_factors))
        return 0.0

    def analyze_pos_distribution(self, pos_tags: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        Analyze part-of-speech distribution.

        Args:
            pos_tags: List of (word, pos) tuples

        Returns:
            Dict of POS -> frequency
        """
        pos_freq = Counter(pos for _, pos in pos_tags)
        return dict(pos_freq)

    def count_repetitions(self, tokens: List[str], window_size: int = 5) -> int:
        """
        Count word repetitions in sliding windows.

        Args:
            tokens: List of tokens
            window_size: Size of sliding window

        Returns:
            Number of repetitions found
        """
        if len(tokens) < 2:
            return 0

        repetition_count = 0
        for i in range(len(tokens) - 1):
            if tokens[i].lower() == tokens[i + 1].lower():
                repetition_count += 1

        # Also check within windows
        for i in range(len(tokens) - window_size):
            window = [t.lower() for t in tokens[i:i+window_size]]
            seen = set()
            for token in window:
                if token in seen:
                    repetition_count += 1
                seen.add(token)

        return repetition_count

    def analyze_pronoun_usage(self, tokens: List[str]) -> Dict[str, int]:
        """
        Analyze pronoun usage patterns.

        Args:
            tokens: List of tokens

        Returns:
            Dict of pronoun_type -> frequency
        """
        pronoun_freq = {
            'first_person': 0,
            'second_person': 0,
            'third_person': 0,
            'demonstrative': 0,
        }

        for token in tokens:
            token_lower = token.lower()
            for pron_type, pronouns in self.PRONOUNS.items():
                if token_lower in pronouns:
                    pronoun_freq[pron_type] += 1

        return pronoun_freq

    def analyze_verb_tenses(self, sentences: List[str]) -> Dict[str, int]:
        """
        Analyze verb tense distribution.

        Args:
            sentences: List of sentences

        Returns:
            Dict of tense -> frequency
        """
        tense_dist = {
            'past': 0,
            'present': 0,
            'future': 0,
            'conditional': 0,
        }

        # Simple pattern matching for common verb forms
        for sent in sentences:
            sent_lower = sent.lower()

            # Past tense markers
            if re.search(r'\b(was|were|did|had|went|came|said)\b', sent_lower):
                tense_dist['past'] += 1

            # Present tense markers
            if re.search(r'\b(is|are|am|have|do|go|come|say)\b', sent_lower):
                tense_dist['present'] += 1

            # Future tense markers
            if re.search(r'\b(will|shall|going to|gonna)\b', sent_lower):
                tense_dist['future'] += 1

            # Conditional markers
            if re.search(r'\b(would|could|should|might)\b', sent_lower):
                tense_dist['conditional'] += 1

        return tense_dist

    def count_named_entities(self, text: str) -> int:
        """
        Count named entities (proper nouns).

        Args:
            text: Text to analyze

        Returns:
            Number of named entities
        """
        if self.nlp is None:
            # Fallback: count capitalized words (heuristic)
            words = text.split()
            return sum(1 for w in words if len(w) > 1 and w[0].isupper())

        try:
            doc = self.nlp(text)
            return len(doc.ents)
        except Exception as e:
            logger.warning(f"Error counting entities: {e}")
            return 0

    def count_clauses(self, sentences: List[str]) -> int:
        """
        Count clauses in text.

        Args:
            sentences: List of sentences

        Returns:
            Approximate clause count
        """
        clause_count = 0

        for sent in sentences:
            # Simple heuristic: count conjunctions and subordinators
            subordinators = ['because', 'since', 'although', 'while', 'when',
                            'if', 'unless', 'after', 'before', 'that', 'which']
            conjunctions = [' and ', ' or ', ' but ', ' nor ', ' yet ']

            sent_lower = ' ' + sent.lower() + ' '
            for marker in subordinators + conjunctions:
                if marker in sent_lower:
                    clause_count += 1

        return clause_count

    def analyze_passive_voice(self, sentences: List[str]) -> float:
        """
        Estimate passive voice ratio.

        Args:
            sentences: List of sentences

        Returns:
            Ratio of passive voice sentences (0-1)
        """
        if not sentences:
            return 0.0

        passive_count = 0
        # Simple pattern: "to be" + past participle
        passive_pattern = r'\b(is|are|was|were|been|be)\s+\w+ed\b'

        for sent in sentences:
            if re.search(passive_pattern, sent.lower()):
                passive_count += 1

        return float(passive_count / len(sentences))

    def analyze(self, text: str, sentences: List[str], tokens: List[str],
               pos_tags: Optional[List[Tuple[str, str]]] = None) -> LinguisticFeatures:
        """
        Complete linguistic analysis.

        Args:
            text: Full text
            sentences: List of sentences
            tokens: List of tokens
            pos_tags: POS tags (optional)

        Returns:
            LinguisticFeatures object
        """
        # Discourse markers
        markers, marker_freq = self.identify_discourse_markers(text)

        # Lexical diversity
        ttr = self.calculate_type_token_ratio(tokens)

        # Word frequency
        word_freq = self.calculate_word_frequency(tokens)

        # POS distribution
        pos_dist = self.analyze_pos_distribution(pos_tags or [])

        # Syntax complexity
        complexity = self.analyze_syntax_complexity(sentences, pos_tags)

        # Sentence statistics
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0

        word_lengths = [len(token) for token in tokens]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0

        # Repetitions
        repetitions = self.count_repetitions(tokens)

        # Discourse coherence (based on marker usage)
        # More discourse markers = better coherence
        marker_density = len(markers) / max(1, len(tokens))
        discourse_coherence = min(1.0, marker_density * 10)

        # Pronouns
        pronouns = self.analyze_pronoun_usage(tokens)

        # Verb tenses
        verb_tenses = self.analyze_verb_tenses(sentences)

        # Named entities
        entities = self.count_named_entities(text)

        # Clauses
        clauses = self.count_clauses(sentences)

        # Passive voice
        passive_ratio = self.analyze_passive_voice(sentences)

        return LinguisticFeatures(
            discourse_markers=markers,
            marker_frequency=marker_freq,
            type_token_ratio=float(ttr),
            word_frequency=word_freq,
            pos_distribution=pos_dist,
            syntax_complexity=float(complexity),
            avg_sentence_length=float(avg_sent_length),
            avg_word_length=float(avg_word_length),
            repetition_count=repetitions,
            discourse_coherence=float(discourse_coherence),
            pronoun_usage=pronouns,
            verb_tense_distribution=verb_tenses,
            named_entity_count=entities,
            clause_count=clauses,
            passive_voice_ratio=float(passive_ratio),
        )
