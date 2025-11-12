"""
Data Cleaner Module

Handles text and audio data cleaning and normalization for preprocessing chat messages.
"""

import re
from typing import Dict, List, Optional


class TextCleaner:
    """Clean and normalize text data from chat messages."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing noise and normalizing.

        Args:
            text: Raw text input from user

        Returns:
            Cleaned text ready for analysis
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-z0-9\s\.\?\!,;:\'\"]', '', text)
        return text
    
    @staticmethod
    def remove_filler_words(text: str) -> str:
        """
        Remove common filler words (um, uh, like, you know, etc.) from speech.

        Args:
            text: Input text

        Returns:
            Text with filler words removed
        """
        fillers = ['um', 'uh', 'like', 'you know', 'i mean', 'basically']
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalize multiple punctuation marks to single marks."""
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        return text
    
    @staticmethod
    def remove_contractions(text: str) -> str:
        """
        Expand contractions for better analysis.
        
        Args:
            text: Input text
            
        Returns:
            Text with contractions expanded
        """
        contractions_dict = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "won't": "will not",
            "wouldn't": "would not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "i'm": "i am",
            "i've": "i have",
            "i'll": "i will",
            "i'd": "i would",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
        }
        
        for contraction, expansion in contractions_dict.items():
            text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
        
        return text


class AudioCleaner:
    """Clean and normalize audio feature data."""
    
    @staticmethod
    def remove_outliers(values: List[float], threshold: float = 3.0) -> List[float]:
        """
        Remove statistical outliers from audio features.
        
        Args:
            values: List of feature values
            threshold: Standard deviations to consider as outlier
            
        Returns:
            List with outliers removed
        """
        if len(values) < 2:
            return values
        
        import numpy as np
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return values
        
        cleaned = [v for v in values if abs((v - mean) / std) <= threshold]
        return cleaned if cleaned else values
    
    @staticmethod
    def smooth_values(values: List[float], window_size: int = 3) -> List[float]:
        """
        Apply moving average smoothing to audio features.
        
        Args:
            values: List of feature values
            window_size: Window size for smoothing
            
        Returns:
            Smoothed values
        """
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            window = values[start:end]
            smoothed.append(sum(window) / len(window))
        
        return smoothed


__all__ = ["TextCleaner", "AudioCleaner"]
