"""
Detection Session Model

Stores individual chat session scores with 12 behavioral parameters.
Each parameter is scored 0-3, max session score = 36.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


def get_time_window_and_session(timestamp: datetime) -> Tuple[str, int]:
    """
    Determine time window and session number from timestamp.

    1 day = 4 sessions:
    - Morning (6:00-11:59): Session 1
    - Afternoon (12:00-15:59): Session 2
    - Evening (16:00-19:59): Session 3
    - Night (20:00-5:59): Session 4

    Args:
        timestamp: Session timestamp

    Returns:
        Tuple of (time_window, session_number)
    """
    hour = timestamp.hour

    if 6 <= hour < 12:
        return "morning", 1
    elif 12 <= hour < 16:
        return "afternoon", 2
    elif 16 <= hour < 20:
        return "evening", 3
    else:  # 20-23 or 0-5
        return "night", 4


class DetectionSessionModel(BaseModel):
    """Detection session data model"""

    # Identifiers
    session_id: str = Field(..., description="Unique session ID")
    user_id: str = Field(default="demo_user", description="User ID (demo_user for testing)")

    # Timestamp and context
    timestamp: datetime = Field(default_factory=datetime.now, description="Session timestamp")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time_window: str = Field(..., description="Time window: morning/afternoon/evening/night")
    session_number: int = Field(..., ge=1, le=4, description="Session number in day (1-4)")

    # 12 Behavioral Parameters (0-3 scale each)
    p1_semantic_incoherence: int = Field(default=0, ge=0, le=3, description="Semantic incoherence")
    p2_repeated_questions: int = Field(default=0, ge=0, le=3, description="Repeated questions")
    p3_self_correction: int = Field(default=0, ge=0, le=3, description="Self-correction frequency")
    p4_low_confidence: int = Field(default=0, ge=0, le=3, description="Low confidence answers")
    p5_hesitation_pauses: int = Field(default=0, ge=0, le=3, description="Hesitation pauses (audio)")
    p6_vocal_tremors: int = Field(default=0, ge=0, le=3, description="Vocal tremors (audio)")
    p7_emotion_slip: int = Field(default=0, ge=0, le=3, description="Emotion + slip")
    p8_slowed_speech: int = Field(default=0, ge=0, le=3, description="Slowed speech (audio)")
    p9_evening_errors: int = Field(default=0, ge=0, le=3, description="Evening errors (time-based)")
    p10_in_session_decline: int = Field(default=0, ge=0, le=3, description="In-session decline")
    p11_memory_recall_failure: int = Field(default=0, ge=0, le=3, description="Memory recall failure")
    p12_topic_maintenance: int = Field(default=0, ge=0, le=3, description="Topic maintenance failure")

    # Session scores
    session_raw_score: int = Field(default=0, ge=0, le=36, description="Sum of 12 parameters (0-36)")

    # Optional: Random Forest probability (when model is available)
    rf_probability: Optional[float] = Field(default=None, ge=0, le=1, description="RF model probability")

    # Session status and lifecycle
    status: str = Field(default="active", description="Session status: active or finalized")

    # Conversation data (accumulated from multiple chat periods)
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="All messages in this session")
    conversation_context: List[str] = Field(default_factory=list, description="Text content of all messages")

    # Chat periods within this session
    chat_periods: List[Dict[str, Any]] = Field(default_factory=list, description="Different chat periods in session")

    # Session timestamps
    session_start: Optional[datetime] = Field(default=None, description="First message timestamp")
    session_end: Optional[datetime] = Field(default=None, description="Last message timestamp")
    last_message_at: Optional[datetime] = Field(default=None, description="Most recent message time")
    finalized_at: Optional[datetime] = Field(default=None, description="When session was finalized")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_2026_01_03_afternoon",
                "user_id": "demo_user",
                "timestamp": "2026-01-03T14:30:00",
                "date": "2026-01-03",
                "time_window": "afternoon",
                "session_number": 2,
                "p1_semantic_incoherence": 2,
                "p2_repeated_questions": 1,
                "p3_self_correction": 3,
                "p4_low_confidence": 0,
                "p5_hesitation_pauses": 2,
                "p6_vocal_tremors": 1,
                "p7_emotion_slip": 2,
                "p8_slowed_speech": 3,
                "p9_evening_errors": 0,
                "p10_in_session_decline": 2,
                "p11_memory_recall_failure": 1,
                "p12_topic_maintenance": 0,
                "session_raw_score": 17,
                "rf_probability": 0.65
            }
        }


class WeeklyRiskModel(BaseModel):
    """Weekly risk calculation model"""

    user_id: str
    week_start: datetime
    week_end: datetime
    sessions_count: int

    # Weekly metrics
    weekly_avg_score: float = Field(..., description="Average session score for the week")
    weekly_base_score: float = Field(..., description="Normalized to 0-100")

    # Trend vs previous week
    previous_week_avg: Optional[float] = Field(default=None)
    weekly_error_increase: Optional[float] = Field(default=None, description="% increase vs previous week")

    # Final weekly risk (0-100)
    final_weekly_risk: float = Field(..., ge=0, le=100, description="Final risk score (0-100)")
    risk_level: str = Field(..., description="Normal/Mild/Moderate/High/Critical")

    # Time window breakdown
    morning_avg: Optional[float] = Field(default=None)
    afternoon_avg: Optional[float] = Field(default=None)
    evening_avg: Optional[float] = Field(default=None)
    night_avg: Optional[float] = Field(default=None)

    # Random Forest validation (optional)
    rf_weekly_avg: Optional[float] = Field(default=None)

    # Metadata
    calculated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "demo_user",
                "week_start": "2026-01-03T00:00:00",
                "week_end": "2026-01-09T23:59:59",
                "sessions_count": 23,
                "weekly_avg_score": 28.3,
                "weekly_base_score": 78.6,
                "previous_week_avg": 24.1,
                "weekly_error_increase": 5.2,
                "final_weekly_risk": 82.7,
                "risk_level": "High",
                "morning_avg": 24.5,
                "afternoon_avg": 26.8,
                "evening_avg": 30.2,
                "night_avg": 32.1,
                "rf_weekly_avg": 0.74
            }
        }


# Database helper functions for MongoDB
class DetectionSessionDB:
    """Database operations for detection sessions"""

    @staticmethod
    async def save_session(db, session_data: Dict[str, Any]) -> str:
        """
        Save detection session to MongoDB.

        Args:
            db: MongoDB database instance
            session_data: Session data dictionary

        Returns:
            Inserted document ID
        """
        try:
            collection = db["chat_risk_predictions"]
            result = await collection.insert_one(session_data)
            logger.info(f"Session saved: {session_data['session_id']}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise

    @staticmethod
    async def get_sessions_by_user(
        db,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve sessions for a user within a date range.

        Args:
            db: MongoDB database instance
            user_id: User ID
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of session documents
        """
        try:
            collection = db["chat_risk_predictions"]

            query = {"user_id": user_id}

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort("timestamp", 1)
            sessions = await cursor.to_list(length=None)

            logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")
            return sessions

        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            raise

    @staticmethod
    async def get_session_by_id(db, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session by ID"""
        try:
            collection = db["chat_risk_predictions"]
            session = await collection.find_one({"session_id": session_id})
            return session
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            raise

    @staticmethod
    async def get_or_create_session(
        db,
        session_id: str,
        user_id: str,
        date: str,
        time_window: str,
        session_number: int,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Get existing session or create new one.
        This enables session accumulation across multiple chat periods.

        Args:
            db: MongoDB database instance
            session_id: Unique session identifier
            user_id: User ID
            date: Date in YYYY-MM-DD format
            time_window: Time window (morning/afternoon/evening/night)
            session_number: Session number (1-4)
            timestamp: Current message timestamp

        Returns:
            Session document (existing or newly created)
        """
        try:
            collection = db["chat_risk_predictions"]

            # Try to find existing session
            existing_session = await collection.find_one({"session_id": session_id})

            if existing_session:
                logger.info(f"Found existing session: {session_id}")
                return existing_session

            # Create new session
            new_session = {
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "date": date,
                "time_window": time_window,
                "session_number": session_number,

                # Initialize all 12 parameters to 0
                "p1_semantic_incoherence": 0,
                "p2_repeated_questions": 0,
                "p3_self_correction": 0,
                "p4_low_confidence": 0,
                "p5_hesitation_pauses": 0,
                "p6_vocal_tremors": 0,
                "p7_emotion_slip": 0,
                "p8_slowed_speech": 0,
                "p9_evening_errors": 0,
                "p10_in_session_decline": 0,
                "p11_memory_recall_failure": 0,
                "p12_topic_maintenance": 0,

                # Initialize session score
                "session_raw_score": 0,
                "rf_probability": None,

                # Session status
                "status": "active",

                # Conversation data
                "messages": [],
                "conversation_context": [],
                "chat_periods": [],

                # Timestamps
                "session_start": timestamp,
                "session_end": None,
                "last_message_at": timestamp,
                "finalized_at": None,

                "created_at": datetime.now()
            }

            # Insert new session
            await collection.insert_one(new_session)
            logger.info(f"Created new session: {session_id}")

            return new_session

        except Exception as e:
            logger.error(f"Error getting/creating session: {e}")
            raise

    @staticmethod
    async def update_session(
        db,
        session_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Update an existing session.

        Args:
            db: MongoDB database instance
            session_id: Session ID to update
            update_data: Data to update

        Returns:
            True if successful
        """
        try:
            collection = db["chat_risk_predictions"]

            result = await collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Session updated: {session_id}")
                return True
            else:
                logger.warning(f"No changes made to session: {session_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating session: {e}")
            raise

    @staticmethod
    async def append_message_to_session(
        db,
        session_id: str,
        message_data: Dict[str, Any],
        text: str,
        timestamp: datetime
    ) -> bool:
        """
        Append a new message to an existing session.

        Args:
            db: MongoDB database instance
            session_id: Session ID
            message_data: Message data to append
            text: Message text content
            timestamp: Message timestamp

        Returns:
            True if successful
        """
        try:
            collection = db["chat_risk_predictions"]

            result = await collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "messages": message_data,
                        "conversation_context": text
                    },
                    "$set": {
                        "last_message_at": timestamp,
                        "session_end": timestamp
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"Message appended to session: {session_id}")
                return True
            else:
                logger.warning(f"Failed to append message to session: {session_id}")
                return False

        except Exception as e:
            logger.error(f"Error appending message: {e}")
            raise

    @staticmethod
    async def finalize_session(
        db,
        session_id: str,
        final_scores: Dict[str, int],
        session_raw_score: int
    ) -> bool:
        """
        Finalize a session (mark as complete).

        Args:
            db: MongoDB database instance
            session_id: Session ID to finalize
            final_scores: Final parameter scores (p1-p12)
            session_raw_score: Final session raw score

        Returns:
            True if successful
        """
        try:
            collection = db["chat_risk_predictions"]

            update_data = {
                "status": "finalized",
                "finalized_at": datetime.now(),
                "session_raw_score": session_raw_score,
                **final_scores
            }

            result = await collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Session finalized: {session_id}, score: {session_raw_score}/36")
                return True
            else:
                logger.warning(f"Failed to finalize session: {session_id}")
                return False

        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            raise

    @staticmethod
    async def get_active_sessions(db) -> List[Dict[str, Any]]:
        """
        Get all active (non-finalized) sessions.

        Args:
            db: MongoDB database instance

        Returns:
            List of active session documents
        """
        try:
            collection = db["chat_risk_predictions"]

            cursor = collection.find({"status": "active"})
            sessions = await cursor.to_list(length=None)

            logger.info(f"Retrieved {len(sessions)} active sessions")
            return sessions

        except Exception as e:
            logger.error(f"Error retrieving active sessions: {e}")
            raise

    @staticmethod
    async def create_indexes(db):
        """Create indexes for detection_sessions collection"""
        try:
            collection = db["chat_risk_predictions"]

            # Create indexes
            await collection.create_index("session_id", unique=True)
            await collection.create_index("user_id")
            await collection.create_index("timestamp")
            await collection.create_index([("user_id", 1), ("timestamp", -1)])
            await collection.create_index("time_window")
            await collection.create_index("status")
            await collection.create_index([("status", 1), ("last_message_at", -1)])

            logger.info("Detection session indexes created")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
