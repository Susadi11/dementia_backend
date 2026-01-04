"""
Session Finalizer Background Job

Automatically finalizes sessions when their time windows end or after inactivity.
Runs every hour to check for sessions that need finalization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from src.database import Database
from src.models.detection_session import DetectionSessionDB
from src.services.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)


class SessionFinalizer:
    """
    Background service to finalize active sessions.

    Finalization triggers:
    1. Time window has ended (e.g., morning ends at 12:00)
    2. 2+ hours of inactivity
    3. End of day (midnight)
    """

    def __init__(self):
        """Initialize session finalizer"""
        self.scoring_engine = ScoringEngine()
        self.is_running = False

    def should_finalize_session(
        self,
        session: Dict[str, Any],
        current_time: datetime
    ) -> tuple[bool, str]:
        """
        Determine if a session should be finalized.

        Args:
            session: Session document
            current_time: Current datetime

        Returns:
            Tuple of (should_finalize, reason)
        """
        time_window = session.get("time_window")
        last_message_at = session.get("last_message_at")
        session_date = session.get("date")

        current_hour = current_time.hour
        current_date = current_time.strftime("%Y-%m-%d")

        # Reason 1: Time window has ended
        if time_window == "morning" and current_hour >= 12:
            return True, "Morning window ended (12:00 PM)"

        elif time_window == "afternoon" and current_hour >= 16:
            return True, "Afternoon window ended (4:00 PM)"

        elif time_window == "evening" and current_hour >= 20:
            return True, "Evening window ended (8:00 PM)"

        elif time_window == "night" and current_hour >= 6:
            return True, "Night window ended (6:00 AM)"

        # Reason 2: 2+ hours of inactivity
        if last_message_at:
            inactivity_hours = (current_time - last_message_at).total_seconds() / 3600
            if inactivity_hours >= 2:
                return True, f"Inactivity timeout ({inactivity_hours:.1f} hours)"

        # Reason 3: Next day has started
        if session_date != current_date:
            return True, "Next day started"

        return False, ""

    async def finalize_session_with_scores(
        self,
        db,
        session: Dict[str, Any]
    ) -> bool:
        """
        Finalize a session by calculating final scores.

        Args:
            db: MongoDB database instance
            session: Session document

        Returns:
            True if successful
        """
        try:
            session_id = session.get("session_id")
            conversation_context = session.get("conversation_context", [])

            if not conversation_context:
                logger.warning(f"Session {session_id} has no messages, finalizing with score 0")
                final_scores = {f"p{i}_" + name: 0 for i, name in enumerate([
                    "semantic_incoherence", "repeated_questions", "self_correction",
                    "low_confidence", "hesitation_pauses", "vocal_tremors",
                    "emotion_slip", "slowed_speech", "evening_errors",
                    "in_session_decline", "memory_recall_failure", "topic_maintenance"
                ], 1)}
                session_raw_score = 0
            else:
                # Combine all messages for final analysis
                all_text = " ".join(conversation_context)
                last_message_time = session.get("last_message_at") or datetime.now()

                # Calculate final scores
                self.scoring_engine.conversation_history = conversation_context
                analysis_result = self.scoring_engine.analyze_session(
                    text=all_text,
                    audio_features=None,
                    timestamp=last_message_time,
                    conversation_context=conversation_context
                )

                scores = analysis_result["scores"]
                session_raw_score = analysis_result["session_raw_score"]

                final_scores = {
                    "p1_semantic_incoherence": scores["p1_semantic_incoherence"],
                    "p2_repeated_questions": scores["p2_repeated_questions"],
                    "p3_self_correction": scores["p3_self_correction"],
                    "p4_low_confidence": scores["p4_low_confidence"],
                    "p5_hesitation_pauses": scores["p5_hesitation_pauses"],
                    "p6_vocal_tremors": scores["p6_vocal_tremors"],
                    "p7_emotion_slip": scores["p7_emotion_slip"],
                    "p8_slowed_speech": scores["p8_slowed_speech"],
                    "p9_evening_errors": scores["p9_evening_errors"],
                    "p10_in_session_decline": scores["p10_in_session_decline"],
                    "p11_memory_recall_failure": scores["p11_memory_recall_failure"],
                    "p12_topic_maintenance": scores["p12_topic_maintenance"]
                }

            # Finalize session in database
            success = await DetectionSessionDB.finalize_session(
                db=db,
                session_id=session_id,
                final_scores=final_scores,
                session_raw_score=session_raw_score
            )

            if success:
                logger.info(
                    f"Session finalized: {session_id}, "
                    f"final score: {session_raw_score}/36, "
                    f"messages: {len(conversation_context)}"
                )

            return success

        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            return False

    async def run_finalization_check(self):
        """
        Check all active sessions and finalize those that meet criteria.
        This should be called periodically (e.g., every hour).
        """
        try:
            db = Database.db
            if not db:
                logger.warning("Database not connected, skipping finalization check")
                return

            current_time = datetime.now()
            logger.info(f"Running session finalization check at {current_time}")

            # Get all active sessions
            active_sessions = await DetectionSessionDB.get_active_sessions(db)

            if not active_sessions:
                logger.info("No active sessions to check")
                return

            logger.info(f"Checking {len(active_sessions)} active sessions")

            finalized_count = 0
            for session in active_sessions:
                session_id = session.get("session_id")

                # Check if should finalize
                should_finalize, reason = self.should_finalize_session(session, current_time)

                if should_finalize:
                    logger.info(f"Finalizing session {session_id}: {reason}")
                    success = await self.finalize_session_with_scores(db, session)
                    if success:
                        finalized_count += 1

            logger.info(f"Finalization check complete. Finalized {finalized_count} sessions.")

        except Exception as e:
            logger.error(f"Error in finalization check: {e}")

    async def start_background_task(self, interval_minutes: int = 60):
        """
        Start background task that runs finalization checks periodically.

        Args:
            interval_minutes: How often to run checks (default: 60 minutes)
        """
        self.is_running = True
        logger.info(f"Starting session finalizer (runs every {interval_minutes} minutes)")

        while self.is_running:
            try:
                await self.run_finalization_check()

                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in background task: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def stop_background_task(self):
        """Stop the background task"""
        logger.info("Stopping session finalizer")
        self.is_running = False


# Global instance
session_finalizer = SessionFinalizer()
