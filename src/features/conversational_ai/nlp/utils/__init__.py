"""NLP Utilities module"""

from .text_preprocessing import (
    TextPreprocessor,
    PreprocessedText,
    clean_text,
    get_ngrams,
    calculate_lexical_diversity,
    identify_repetitions,
)

__all__ = [
    'TextPreprocessor',
    'PreprocessedText',
    'clean_text',
    'get_ngrams',
    'calculate_lexical_diversity',
    'identify_repetitions',
]
