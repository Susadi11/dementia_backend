"""
Configuration Module for Dementia Detection System

This module handles all configuration settings for the dementia detection system,
including paths, feature extraction parameters, and system settings.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class PathConfig:
    """
    Configuration for file paths and directories.

    Attributes:
        data_dir: Root directory containing all datasets
        output_dir: Directory for processed data and results
        models_dir: Directory for saved ML models
        logs_dir: Directory for log files
        cache_dir: Directory for cached intermediate results
    """
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "./models")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", "./logs")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction parameters for dementia detection.

    The 10 key parameters:
    1. Semantic incoherence
    2. Repeated questions
    3. Self-correction
    4. Low-confidence answer
    5. Hesitation pauses
    6. Vocal tremors
    7. Emotion + slip
    8. Slowed speech
    9. Evening errors
    10. In-session decline

    Attributes:
        audio_sample_rate: Sample rate for audio processing
        audio_chunk_duration: Duration of audio chunks in seconds
        text_min_words: Minimum number of words for valid analysis
        confidence_threshold: Threshold for low-confidence detection
        hesitation_threshold: Minimum pause duration to count as hesitation (ms)
        speech_rate_threshold: Threshold for slowed speech detection (words/min)
    """
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_chunk_duration: int = int(os.getenv("AUDIO_CHUNK_DURATION", "30"))
    text_min_words: int = int(os.getenv("TEXT_MIN_WORDS", "5"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    hesitation_threshold: int = int(os.getenv("HESITATION_THRESHOLD", "500"))
    speech_rate_threshold: int = int(os.getenv("SPEECH_RATE_THRESHOLD", "120"))
    enable_voice_analysis: bool = os.getenv("ENABLE_VOICE_ANALYSIS", "True").lower() == "true"
    enable_text_analysis: bool = os.getenv("ENABLE_TEXT_ANALYSIS", "True").lower() == "true"


@dataclass
class ProcessingConfig:
    """
    Configuration for data processing and system settings.

    Attributes:
        n_jobs: Number of parallel jobs for multiprocessing
        batch_size: Batch size for batch processing
        cache_enabled: Whether to enable caching of processed data
        verbose: Verbosity level (0=silent, 1=progress bars, 2=detailed)
        max_audio_length: Maximum audio length in seconds
        supported_audio_formats: List of supported audio file formats
    """
    n_jobs: int = int(os.getenv("N_JOBS", "4"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    verbose: int = 1
    max_audio_length: int = int(os.getenv("MAX_AUDIO_LENGTH", "600"))
    supported_audio_formats: list = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg"])


@dataclass
class LoggingConfig:
    """
    Configuration for logging settings.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        rotation: Log file rotation settings (e.g., "500 MB")
        retention: Log file retention period (e.g., "10 days")
        log_file: Log file name
    """
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv(
        "LOG_FORMAT",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    rotation: str = "500 MB"
    retention: str = "10 days"
    log_file: str = "dementia_detection.log"


@dataclass
class APIConfig:
    """
    Configuration for API server settings.

    Attributes:
        host: Server host address
        port: Server port
        debug: Debug mode
        cors_origins: Allowed CORS origins
        max_upload_size: Maximum file upload size in MB
        api_title: API documentation title
        api_version: API version
    """
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "True").lower() == "true"
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "100"))  # MB
    api_title: str = "Dementia Detection API"
    api_version: str = "1.0.0"
    api_description: str = "API for detecting dementia using conversational AI analysis"


@dataclass
class ModelConfig:
    """
    Configuration for machine learning model settings.

    Attributes:
        default_model: Default model to use for predictions
        model_threshold: Threshold for positive classification
        ensemble_voting: Voting strategy for ensemble models
        cross_validation_folds: Number of folds for cross-validation
    """
    default_model: str = os.getenv("DEFAULT_MODEL", "random_forest")
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.5"))
    ensemble_voting: str = "soft"
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class NLPConfig:
    """
    Configuration for NLP processing and analysis.

    Handles text processing, semantic analysis, and feature extraction
    for dementia marker detection.

    Attributes:
        enabled: Enable NLP processing module
        spacy_model: SpaCy model name (e.g., en_core_web_sm)
        semantic_model: Transformer model for semantic analysis
        use_sentence_transformers: Use SentenceTransformers for efficiency
        enable_semantic: Enable semantic coherence analysis
        enable_emotion: Enable sentiment/emotion analysis
        enable_linguistic: Enable linguistic feature extraction
        device: Device to use ('cpu' or 'cuda')
        cache_models: Cache loaded models in memory
        remove_stopwords: Remove stopwords during preprocessing
        include_embeddings: Include embedding vectors in output (memory intensive)
        batch_processing: Enable batch processing for multiple texts
    """
    enabled: bool = os.getenv("NLP_ENABLED", "True").lower() == "true"
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    semantic_model: str = os.getenv("SEMANTIC_MODEL", "distilbert-base-uncased")
    use_sentence_transformers: bool = os.getenv("USE_SENTENCE_TRANSFORMERS", "True").lower() == "true"
    enable_semantic: bool = os.getenv("NLP_ENABLE_SEMANTIC", "True").lower() == "true"
    enable_emotion: bool = os.getenv("NLP_ENABLE_EMOTION", "True").lower() == "true"
    enable_linguistic: bool = os.getenv("NLP_ENABLE_LINGUISTIC", "True").lower() == "true"
    device: str = os.getenv("NLP_DEVICE", "cpu")
    cache_models: bool = os.getenv("NLP_CACHE_MODELS", "True").lower() == "true"
    remove_stopwords: bool = os.getenv("NLP_REMOVE_STOPWORDS", "False").lower() == "true"
    include_embeddings: bool = os.getenv("NLP_INCLUDE_EMBEDDINGS", "False").lower() == "true"
    batch_processing: bool = os.getenv("NLP_BATCH_PROCESSING", "True").lower() == "true"


class Config:
    """
    Main configuration class that aggregates all configuration settings.

    This class provides a centralized access point for all configuration
    parameters used throughout the application.
    """

    def __init__(self):
        """Initialize all configuration sections."""
        self.paths = PathConfig()
        self.features = FeatureConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        self.api = APIConfig()
        self.model = ModelConfig()
        self.nlp = NLPConfig()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            "paths": self.paths.__dict__,
            "features": self.features.__dict__,
            "processing": self.processing.__dict__,
            "logging": self.logging.__dict__,
            "api": self.api.__dict__,
            "model": self.model.__dict__,
            "nlp": self.nlp.__dict__,
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        if self.features.confidence_threshold < 0 or self.features.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if self.model.model_threshold < 0 or self.model.model_threshold > 1:
            raise ValueError("Model threshold must be between 0 and 1")

        if self.processing.n_jobs < 1:
            raise ValueError("Number of jobs must be at least 1")

        return True

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(paths={self.paths}, features={self.features}, api={self.api})"


# Create global configuration instance
config = Config()


# Export commonly used configurations
__all__ = [
    "config",
    "PathConfig",
    "FeatureConfig",
    "ProcessingConfig",
    "LoggingConfig",
    "APIConfig",
    "ModelConfig",
    "NLPConfig",
]
