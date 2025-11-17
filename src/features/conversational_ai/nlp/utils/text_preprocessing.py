"""
Text Preprocessing Module

Handles:
- Text normalization (lowercase, special symbols removal, punctuation cleanup)
- Tokenization and lemmatization using spaCy
- Language detection
- Contraction expansion
- Whitespace normalization
"""

import re
import logging
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    LANGDETECT_AVAILABLE = True
    DetectorFactory.seed = 0  # For reproducibility
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedText:
    """Container for preprocessed text data"""
    original_text: str
    normalized_text: str
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[Tuple[str, str]]
    language: str
    word_count: int
    unique_words: int
    avg_word_length: float
    sentences: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'original_text': self.original_text,
            'normalized_text': self.normalized_text,
            'tokens': self.tokens,
            'lemmas': self.lemmas,
            'pos_tags': self.pos_tags,
            'language': self.language,
            'word_count': self.word_count,
            'unique_words': self.unique_words,
            'avg_word_length': self.avg_word_length,
            'sentences': self.sentences,
        }


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.

    Performs normalization, tokenization, lemmatization, and linguistic analysis.
    """

    # Common contractions dictionary
    CONTRACTIONS = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }

    # Filler words and discourse markers to identify hesitations
    FILLERS = {
        'um', 'uh', 'er', 'erm', 'ah', 'eh', 'hmm', 'mm', 'hmph',
        'like', 'you know', 'i mean', 'basically', 'literally', 'honestly'
    }

    def __init__(self, spacy_model: str = "en_core_web_sm",
                 enable_lemmatization: bool = True,
                 enable_language_detection: bool = True):
        """
        Initialize TextPreprocessor.

        Args:
            spacy_model: SpaCy model to load (default: en_core_web_sm)
            enable_lemmatization: Whether to perform lemmatization
            enable_language_detection: Whether to detect language
        """
        self.enable_lemmatization = enable_lemmatization and SPACY_AVAILABLE
        self.enable_language_detection = enable_language_detection and LANGDETECT_AVAILABLE
        self.nlp = None

        # Load spaCy model if available and enabled
        if self.enable_lemmatization:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"SpaCy model '{spacy_model}' not found. "
                             f"Run: python -m spacy download {spacy_model}")
                self.enable_lemmatization = False

        # Initialize NLTK stopwords if available
        self.stopwords = set()
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('english'))
            except LookupError:
                logger.warning("NLTK stopwords not found. "
                             "Run: nltk.download('stopwords')")

    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove special chars, clean punctuation.

        Args:
            text: Raw input text

        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        text = self._expand_contractions(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep basic punctuation for now
        # (we'll handle punctuation more carefully)
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)

        # Normalize multiple punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\!+', '!', text)
        text = re.sub(r'\?+', '?', text)

        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        pattern = re.compile(r'\b(' + '|'.join(self.CONTRACTIONS.keys()) + r')\b')
        return pattern.sub(lambda x: self.CONTRACTIONS[x.group(0)], text)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()

        # Further split on punctuation if needed
        detailed_tokens = []
        for token in tokens:
            # Keep punctuation attached to words for now
            detailed_tokens.append(token)

        return detailed_tokens

    def lemmatize(self, tokens: List[str], original_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Lemmatize tokens using spaCy.

        Args:
            tokens: List of tokens
            original_text: Original text for context

        Returns:
            Tuple of (lemmas, pos_tags)
        """
        if not self.enable_lemmatization or self.nlp is None:
            # Fallback: return tokens as-is with basic POS tagging
            return tokens, [(token, 'NOUN') for token in tokens]

        try:
            doc = self.nlp(original_text)
            lemmas = [token.lemma_ for token in doc]
            pos_tags = [(token.text, token.pos_) for token in doc]
            return lemmas, pos_tags
        except Exception as e:
            logger.warning(f"Lemmatization error: {e}")
            return tokens, [(token, 'NOUN') for token in tokens]

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        if not self.enable_language_detection:
            return 'en'  # Default to English

        try:
            language = detect(text)
            return language
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return 'en'  # Default to English on error

    def identify_sentences(self, text: str) -> List[str]:
        """
        Identify sentences in text.

        Args:
            text: Text to analyze

        Returns:
            List of sentences
        """
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                return [sent.text for sent in doc.sents]
            except Exception as e:
                logger.warning(f"Sentence segmentation error: {e}")

        # Fallback: simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def identify_fillers(self, tokens: List[str]) -> List[Tuple[int, str]]:
        """
        Identify filler words (hesitation markers).

        Args:
            tokens: List of tokens

        Returns:
            List of (index, filler_word) tuples
        """
        fillers = []
        for idx, token in enumerate(tokens):
            # Check exact matches
            if token.lower() in self.FILLERS:
                fillers.append((idx, token))
            # Check multi-word fillers
            elif idx < len(tokens) - 1:
                two_word = f"{token.lower()} {tokens[idx+1].lower()}"
                if two_word in self.FILLERS:
                    fillers.append((idx, two_word))

        return fillers

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation tokens."""
        return [token for token in tokens
                if token not in string.punctuation and token.strip()]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        if not self.stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stopwords]

    def process(self, text: str,
                remove_stopwords: bool = False,
                remove_punctuation: bool = True) -> PreprocessedText:
        """
        Complete preprocessing pipeline.

        Args:
            text: Raw input text
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation tokens

        Returns:
            PreprocessedText object with all extracted features
        """
        # Normalize
        normalized = self.normalize_text(text)

        # Tokenize
        tokens = self.tokenize(normalized)

        # Remove punctuation if requested
        if remove_punctuation:
            tokens = self.remove_punctuation(tokens)

        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Lemmatize
        lemmas, pos_tags = self.lemmatize(tokens, text)

        # Detect language
        language = self.detect_language(text)

        # Identify sentences
        sentences = self.identify_sentences(text)

        # Calculate statistics
        word_count = len(tokens)
        unique_words = len(set(tokens))
        avg_word_length = sum(len(token) for token in tokens) / max(word_count, 1)

        return PreprocessedText(
            original_text=text,
            normalized_text=normalized,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            language=language,
            word_count=word_count,
            unique_words=unique_words,
            avg_word_length=avg_word_length,
            sentences=sentences,
        )


# Utility functions
def clean_text(text: str) -> str:
    """Quick text cleaning (lightweight alternative)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_lexical_diversity(tokens: List[str]) -> float:
    """Calculate Type-Token Ratio (TTR) for lexical diversity."""
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def identify_repetitions(tokens: List[str], window_size: int = 5) -> List[Tuple[int, str]]:
    """
    Identify repeated words within a sliding window.

    Args:
        tokens: List of tokens
        window_size: Size of sliding window

    Returns:
        List of (position, repeated_word) tuples
    """
    repetitions = []
    for i in range(len(tokens) - window_size):
        window = tokens[i:i+window_size]
        # Count occurrences of each word in window
        seen = {}
        for j, token in enumerate(window):
            token_lower = token.lower()
            if token_lower not in seen:
                seen[token_lower] = j
            else:
                # Found a repetition
                repetitions.append((i + j, token))

    return repetitions
