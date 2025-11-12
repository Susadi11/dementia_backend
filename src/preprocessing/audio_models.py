"""
Database Models for Audio and Transcript Storage

Handles persistence of audio file metadata, transcripts, analysis results, and relationships.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class AudioRecord(Base):
    """Stores audio file metadata and transcripts."""

    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    session_id = Column(String(255), index=True, nullable=False)
    original_filename = Column(String(512), nullable=False)
    audio_path = Column(String(1024), nullable=False)
    audio_format = Column(String(50), default="wav")
    audio_size_bytes = Column(Integer)
    duration_seconds = Column(Float)
    transcript = Column(Text)
    transcript_path = Column(String(1024))
    language_detected = Column(String(10), default="en")
    transcription_confidence = Column(Float, default=0.0)
    noise_level = Column(Float, default=0.0)
    is_enhanced = Column(Boolean, default=True)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_model = Column(String(100), default="whisper-base")
    processing_status = Column(String(50), default="completed")
    processing_error = Column(Text)
    analyzed = Column(Boolean, default=False)
    analysis_timestamp = Column(DateTime)

    def __repr__(self):
        return f"<AudioRecord(user_id={self.user_id}, session_id={self.session_id}, duration={self.duration_seconds}s)>"

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'original_filename': self.original_filename,
            'audio_path': self.audio_path,
            'audio_format': self.audio_format,
            'audio_size_bytes': self.audio_size_bytes,
            'duration_seconds': self.duration_seconds,
            'transcript': self.transcript,
            'transcript_path': self.transcript_path,
            'language_detected': self.language_detected,
            'transcription_confidence': self.transcription_confidence,
            'noise_level': self.noise_level,
            'is_enhanced': self.is_enhanced,
            'processing_timestamp': self.processing_timestamp.isoformat() if self.processing_timestamp else None,
            'processing_model': self.processing_model,
            'processing_status': self.processing_status,
            'processing_error': self.processing_error,
            'analyzed': self.analyzed,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None
        }


class TranscriptSegment(Base):
    """Stores detailed transcript segments from Whisper."""

    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, index=True)
    audio_record_id = Column(Integer, index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    session_id = Column(String(255), index=True, nullable=False)
    segment_index = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)
    text = Column(Text)
    confidence = Column(Float, default=0.0)
    contains_hesitation = Column(Boolean, default=False)
    contains_repetition = Column(Boolean, default=False)
    pause_duration = Column(Float, default=0.0)
    speech_rate = Column(Float)
    extracted_timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TranscriptSegment(audio_record_id={self.audio_record_id}, text={self.text[:50]}...)>"

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'audio_record_id': self.audio_record_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'segment_index': self.segment_index,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text,
            'confidence': self.confidence,
            'contains_hesitation': self.contains_hesitation,
            'contains_repetition': self.contains_repetition,
            'pause_duration': self.pause_duration,
            'speech_rate': self.speech_rate,
            'extracted_timestamp': self.extracted_timestamp.isoformat() if self.extracted_timestamp else None
        }


class VoiceAnalysisResult(Base):
    """Stores voice analysis results linking audio to dementia risk scores."""

    __tablename__ = "voice_analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    audio_record_id = Column(Integer, index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    session_id = Column(String(255), index=True, nullable=False)
    semantic_incoherence = Column(Float, default=0.0)
    repeated_questions = Column(Float, default=0.0)
    self_correction = Column(Float, default=0.0)
    low_confidence_answer = Column(Float, default=0.0)
    hesitation_pauses = Column(Float, default=0.0)
    vocal_tremors = Column(Float, default=0.0)
    emotion_slip = Column(Float, default=0.0)
    slowed_speech = Column(Float, default=0.0)
    evening_errors = Column(Float, default=0.0)
    in_session_decline = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(50), default="low")
    risk_description = Column(Text)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    analysis_model_version = Column(String(100))
    notes = Column(Text)

    def __repr__(self):
        return f"<VoiceAnalysisResult(user_id={self.user_id}, risk_score={self.risk_score})>"

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'audio_record_id': self.audio_record_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'semantic_incoherence': self.semantic_incoherence,
            'repeated_questions': self.repeated_questions,
            'self_correction': self.self_correction,
            'low_confidence_answer': self.low_confidence_answer,
            'hesitation_pauses': self.hesitation_pauses,
            'vocal_tremors': self.vocal_tremors,
            'emotion_slip': self.emotion_slip,
            'slowed_speech': self.slowed_speech,
            'evening_errors': self.evening_errors,
            'in_session_decline': self.in_session_decline,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'risk_description': self.risk_description,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'analysis_model_version': self.analysis_model_version,
            'notes': self.notes
        }


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: str = "sqlite:///./dementia_detection.db"):
        """Initialize database connection and create tables."""
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized: {database_url}")

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def save_audio_record(
        self,
        user_id: str,
        session_id: str,
        original_filename: str,
        audio_path: str,
        transcript: str,
        transcript_path: str,
        duration: float,
        language: str,
        confidence: float,
        audio_format: str = "wav",
        audio_size_bytes: Optional[int] = None,
        processing_model: str = "whisper-base"
    ) -> Optional[AudioRecord]:
        """Save audio record with transcript metadata to database."""
        try:
            db = self.get_session()
            record = AudioRecord(
                user_id=user_id,
                session_id=session_id,
                original_filename=original_filename,
                audio_path=audio_path,
                audio_format=audio_format,
                audio_size_bytes=audio_size_bytes,
                duration_seconds=duration,
                transcript=transcript,
                transcript_path=transcript_path,
                language_detected=language,
                transcription_confidence=confidence,
                is_enhanced=True,
                processing_model=processing_model,
                processing_status="completed"
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(f"Saved audio record: {record.id}")
            db.close()
            return record
        except Exception as e:
            logger.error(f"Failed to save audio record: {str(e)}")
            return None

    def save_voice_analysis(
        self,
        audio_record_id: int,
        user_id: str,
        session_id: str,
        features: dict,
        risk_score: float,
        risk_level: str,
        risk_description: str,
        model_version: str = "1.0.0"
    ) -> Optional[VoiceAnalysisResult]:
        """Save voice analysis results with all dementia indicators to database."""
        try:
            db = self.get_session()
            result = VoiceAnalysisResult(
                audio_record_id=audio_record_id,
                user_id=user_id,
                session_id=session_id,
                semantic_incoherence=features.get('semantic_incoherence', 0.0),
                repeated_questions=features.get('repeated_questions', 0.0),
                self_correction=features.get('self_correction', 0.0),
                low_confidence_answer=features.get('low_confidence_answer', 0.0),
                hesitation_pauses=features.get('hesitation_pauses', 0.0),
                vocal_tremors=features.get('vocal_tremors', 0.0),
                emotion_slip=features.get('emotion_slip', 0.0),
                slowed_speech=features.get('slowed_speech', 0.0),
                evening_errors=features.get('evening_errors', 0.0),
                in_session_decline=features.get('in_session_decline', 0.0),
                risk_score=risk_score,
                risk_level=risk_level,
                risk_description=risk_description,
                analysis_model_version=model_version
            )
            db.add(result)
            db.commit()
            db.refresh(result)
            logger.info(f"Saved voice analysis: {result.id}")
            db.close()
            return result
        except Exception as e:
            logger.error(f"Failed to save voice analysis: {str(e)}")
            return None

    def get_user_sessions(self, user_id: str) -> List[AudioRecord]:
        """Retrieve all audio records for a user."""
        try:
            db = self.get_session()
            records = db.query(AudioRecord).filter(
                AudioRecord.user_id == user_id
            ).all()
            db.close()
            return records
        except Exception as e:
            logger.error(f"Failed to retrieve user sessions: {str(e)}")
            return []

    def get_session_audio(self, user_id: str, session_id: str) -> Optional[AudioRecord]:
        """Retrieve audio record for specific user session."""
        try:
            db = self.get_session()
            record = db.query(AudioRecord).filter(
                (AudioRecord.user_id == user_id) &
                (AudioRecord.session_id == session_id)
            ).first()
            db.close()
            return record
        except Exception as e:
            logger.error(f"Failed to retrieve session audio: {str(e)}")
            return None


_db_manager = None


def get_db_manager(database_url: str = "sqlite:///./dementia_detection.db") -> DatabaseManager:
    """Get or create database manager singleton instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager
