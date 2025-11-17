"""
Semantic Analysis Module

Handles:
- BERT/DistilBERT embeddings generation
- Semantic similarity and coherence detection
- Incoherence scoring
- Semantic drift detection
- Topic modeling
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SemanticAnalysis:
    """Container for semantic analysis results"""
    incoherence_score: float  # 0-1, higher = more incoherent
    semantic_coherence: float  # 0-1, higher = more coherent
    semantic_drift: float  # 0-1, degree of topic drift
    avg_similarity: float  # Average similarity between consecutive sentences
    similarity_variance: float  # Variance in similarity scores
    key_topics: List[str]  # Main topics identified
    embeddings: Optional[List[List[float]]]  # Sentence embeddings
    sentence_similarities: List[float]  # Similarity scores between consecutive sentences
    low_coherence_regions: List[Tuple[int, int]]  # (start, end) indices of incoherent regions

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'incoherence_score': float(self.incoherence_score),
            'semantic_coherence': float(self.semantic_coherence),
            'semantic_drift': float(self.semantic_drift),
            'avg_similarity': float(self.avg_similarity),
            'similarity_variance': float(self.similarity_variance),
            'key_topics': self.key_topics,
            'sentence_similarities': [float(s) for s in self.sentence_similarities],
            'low_coherence_regions': self.low_coherence_regions,
        }


class SemanticAnalyzer:
    """
    Semantic analysis using transformer models (BERT, DistilBERT, etc).

    Analyzes semantic coherence, drift, and incoherence in text.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 use_sentence_transformers: bool = True,
                 device: str = "cpu"):
        """
        Initialize SemanticAnalyzer.

        Args:
            model_name: HuggingFace model name
            use_sentence_transformers: Use SentenceTransformers (faster) vs raw transformers
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_sentence_transformers = use_sentence_transformers

        self._load_model()

    def _load_model(self):
        """Load the semantic model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE and not TRANSFORMERS_AVAILABLE:
            logger.warning("Neither SentenceTransformers nor Transformers available. "
                         "Semantic analysis disabled.")
            return

        try:
            if self.use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use SentenceTransformers for faster embeddings
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            elif TRANSFORMERS_AVAILABLE:
                # Use raw transformers
                self.model = AutoModel.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info(f"Loaded Transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading semantic model: {e}")
            self.model = None

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if self.model is None:
            logger.warning("Semantic model not loaded")
            return None

        try:
            if self.use_sentence_transformers and hasattr(self.model, 'encode'):
                # SentenceTransformers
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            else:
                # Raw transformers
                return self._get_embeddings_transformers(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def _get_embeddings_transformers(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using raw transformer model."""
        if self.model is None or self.tokenizer is None:
            return None

        try:
            # Encode all texts
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings
        except Exception as e:
            logger.error(f"Error in transformer embeddings: {e}")
            return None

    def calculate_semantic_coherence(self,
                                   sentences: List[str],
                                   similarity_threshold: float = 0.5) -> SemanticAnalysis:
        """
        Calculate semantic coherence by analyzing sentence similarities.

        Args:
            sentences: List of sentences
            similarity_threshold: Threshold for considering sentences related

        Returns:
            SemanticAnalysis object
        """
        if not sentences or len(sentences) < 2:
            return self._empty_semantic_analysis()

        # Get embeddings
        embeddings = self.get_embeddings(sentences)
        if embeddings is None:
            return self._empty_semantic_analysis()

        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() \
                  if hasattr(util, 'pytorch_cos_sim') else \
                  self._cosine_similarity(embeddings[i], embeddings[i+1])
            # Clamp to [0, 1] range
            sim = max(0, min(1, float(sim)))
            similarities.append(sim)

        # Calculate statistics
        similarities_array = np.array(similarities)
        avg_similarity = float(np.mean(similarities_array)) if len(similarities_array) > 0 else 0.0
        similarity_variance = float(np.var(similarities_array)) if len(similarities_array) > 0 else 0.0

        # Identify low coherence regions
        low_coherence_regions = []
        in_region = False
        region_start = 0
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold and not in_region:
                region_start = i
                in_region = True
            elif sim >= similarity_threshold and in_region:
                low_coherence_regions.append((region_start, i))
                in_region = False
        if in_region:
            low_coherence_regions.append((region_start, len(similarities)))

        # Calculate incoherence score (inverse of coherence)
        # More variance and lower similarity = higher incoherence
        incoherence_score = float(1.0 - avg_similarity + (similarity_variance * 0.1))
        incoherence_score = max(0, min(1, incoherence_score))

        # Semantic drift: how much topics change over time
        semantic_drift = self._calculate_semantic_drift(embeddings)

        # Extract topics
        key_topics = self._extract_topics(sentences, embeddings)

        return SemanticAnalysis(
            incoherence_score=incoherence_score,
            semantic_coherence=1.0 - incoherence_score,
            semantic_drift=semantic_drift,
            avg_similarity=avg_similarity,
            similarity_variance=similarity_variance,
            key_topics=key_topics,
            embeddings=[emb.tolist() for emb in embeddings],
            sentence_similarities=similarities,
            low_coherence_regions=low_coherence_regions,
        )

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def _calculate_semantic_drift(self, embeddings: np.ndarray) -> float:
        """
        Calculate semantic drift - how much the topic shifts over time.

        Uses the distance from the first sentence to subsequent sentences.
        """
        if len(embeddings) < 2:
            return 0.0

        distances = []
        for i in range(1, len(embeddings)):
            dist = 1.0 - self._cosine_similarity(embeddings[0], embeddings[i])
            distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def _extract_topics(self, sentences: List[str], embeddings: np.ndarray,
                       n_topics: int = 3) -> List[str]:
        """
        Extract key topics from sentences.

        Simple approach: return longest sentences with highest variance
        """
        if len(sentences) == 0:
            return []

        # Get sentence importance by embedding variance
        variances = np.var(embeddings, axis=1)
        top_indices = np.argsort(variances)[-n_topics:][::-1]

        topics = []
        for idx in top_indices:
            if idx < len(sentences):
                # Extract main words from sentence (simple heuristic)
                words = sentences[idx].split()
                # Get longest non-common words
                important_words = [w for w in words if len(w) > 4][:2]
                if important_words:
                    topics.append(' '.join(important_words))

        return topics[:n_topics]

    def detect_incoherent_spans(self,
                               sentences: List[str],
                               coherence_threshold: float = 0.4) -> List[Dict]:
        """
        Detect spans of incoherent text.

        Args:
            sentences: List of sentences
            coherence_threshold: Threshold below which text is incoherent

        Returns:
            List of incoherent span dicts with start, end, and reasons
        """
        if len(sentences) < 2:
            return []

        embeddings = self.get_embeddings(sentences)
        if embeddings is None:
            return []

        incoherent_spans = []
        current_span = None

        for i in range(len(embeddings) - 1):
            coherence = self._cosine_similarity(embeddings[i], embeddings[i+1])

            if coherence < coherence_threshold:
                if current_span is None:
                    current_span = {
                        'start': i,
                        'start_text': sentences[i],
                        'coherence_score': coherence,
                        'reason': 'Topic shift detected'
                    }
                else:
                    current_span['end'] = i + 1
                    current_span['end_text'] = sentences[i + 1]
            else:
                if current_span is not None:
                    if 'end' not in current_span:
                        current_span['end'] = i + 1
                        current_span['end_text'] = sentences[i]
                    incoherent_spans.append(current_span)
                    current_span = None

        if current_span is not None:
            current_span['end'] = len(sentences) - 1
            if len(sentences) > 0:
                current_span['end_text'] = sentences[-1]
            incoherent_spans.append(current_span)

        return incoherent_spans

    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embeddings = self.get_embeddings([text1, text2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0

        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        return float(max(0, min(1, similarity)))

    def _empty_semantic_analysis(self) -> SemanticAnalysis:
        """Return empty analysis when computation fails."""
        return SemanticAnalysis(
            incoherence_score=0.0,
            semantic_coherence=1.0,
            semantic_drift=0.0,
            avg_similarity=0.0,
            similarity_variance=0.0,
            key_topics=[],
            embeddings=None,
            sentence_similarities=[],
            low_coherence_regions=[],
        )
