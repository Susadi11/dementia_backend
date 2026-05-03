"""
Reminder System API Routes

FastAPI endpoints for the Context-Aware Smart Reminder System.

Provides endpoints for:
- Creating, updating, and managing reminders
- Processing user responses
- Getting behavior analytics
- Caregiver notifications
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import logging
import uuid
import tempfile
import os
import statistics
from pathlib import Path

from src.features.reminder_system import (
    Reminder, ReminderCommand, ReminderResponse, ReminderInteraction,
    PittBasedReminderAnalyzer, AdaptiveReminderScheduler, BehaviorTracker
)
from src.features.reminder_system.caregiver_notifier import CaregiverNotifier
from src.features.reminder_system.reminder_models import (
    ReminderStatus, ReminderPriority, CaregiverAlert, InteractionType
)

# Behavioral analysis – auto-log every reminder interaction
from src.services.behavioral_timeline_service import BehavioralTimelineService
from src.features.behavioral_analysis.behavioral_models import (
    UserBehavioralLog, ActivityType, ResponseType
)
_behavioral_service: Optional[BehavioralTimelineService] = None

def _get_behavioral_service() -> BehavioralTimelineService:
    global _behavioral_service
    if _behavioral_service is None:
        _behavioral_service = BehavioralTimelineService()
    return _behavioral_service
from src.features.reminder_system.weekly_report_generator import (
    WeeklyReportGenerator, WeeklyCognitiveReport
)
from src.services.reminder_db_service import ReminderDatabaseService
from src.services.user_cognitive_score import UserCognitiveScoreService
from src.database import Database
from src.services.chatbot import get_whisper_service
from src.features.conversational_ai.nlp.nlp_engine import NLPEngine

logger = logging.getLogger(__name__)

# Try to import BERT parser (optional, falls back to regex)
try:
    from src.features.reminder_system.bert_text_parser import get_bert_parser
    BERT_AVAILABLE = True
    logger.info("✅ BERT text parser available")
except ImportError:
    BERT_AVAILABLE = False
    logger.info("⚠️ BERT not available, using regex parser")

# Create router
router = APIRouter(prefix="/api/reminders", tags=["reminders"])

# Initialize components (these would typically come from dependency injection)
reminder_analyzer = PittBasedReminderAnalyzer()
_db_service = ReminderDatabaseService()          # singleton used by get_db_service() too
behavior_tracker = BehaviorTracker(db_service=_db_service)
scheduler = AdaptiveReminderScheduler(behavior_tracker, reminder_analyzer)
caregiver_notifier = CaregiverNotifier()
report_generator = WeeklyReportGenerator(behavior_tracker)
cognitive_score_service = UserCognitiveScoreService(reminder_analyzer, behavior_tracker)
nlp_engine = NLPEngine()
_bert_parser = None

def _parse_reminder_from_text(
    text: str,
    user_id: str,
    priority_override: Optional[str] = None
) -> dict:
    """
    Parse reminder details from natural language text using NLP.
    
    Uses BERT-based parser if available, falls back to regex patterns.
    
    Extracts:
    - Title and description
    - Category (medication, appointment, meal, etc.)
    - Time/date information
    - Recurrence pattern
    - Priority level
    """
    # Try BERT parser first (more accurate)
    if BERT_AVAILABLE:
        try:
            global _bert_parser
            if _bert_parser is None:
                _bert_parser = get_bert_parser()
            
            result = _bert_parser.parse_reminder(text, user_id, priority_override)
            logger.info(f"✅ BERT parsed: {result['category']} at {result['scheduled_time']}")
            return result
        except Exception as e:
            logger.warning(f"BERT parsing failed, falling back to regex: {e}")
    
    # Fallback to regex-based parsing
    return _parse_reminder_regex(text, user_id, priority_override)


def _normalize_spoken_numbers(text: str) -> str:
    """
    Convert spoken number words to digits so time-extraction regexes can match.
    Handles Whisper transcriptions like 'six PM' -> '6 PM', 'three thirty' -> '3:30'.
    """
    import re as _re

    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'twenty-one': '21', 'twenty one': '21', 'twenty-two': '22', 'twenty two': '22',
        'twenty-three': '23', 'twenty three': '23', 'twenty-four': '24', 'twenty four': '24',
        'thirty': '30', 'forty': '40', 'forty-five': '45', 'forty five': '45',
        'fifty': '50',
    }

    result = text.lower()

    # Handle "half past <number>" -> "<number>:30"
    result = _re.sub(
        r'half\s+past\s+(\w+(?:-\w+)?)',
        lambda m: f"{word_to_num.get(m.group(1), m.group(1))}:30",
        result
    )
    # Handle "quarter past <number>" -> "<number>:15"
    result = _re.sub(
        r'quarter\s+past\s+(\w+(?:-\w+)?)',
        lambda m: f"{word_to_num.get(m.group(1), m.group(1))}:15",
        result
    )
    # Handle "quarter to <number>" -> "<number-1>:45"
    def _quarter_to(m):
        w = m.group(1)
        h = int(word_to_num.get(w, w)) if not w.isdigit() else int(w)
        return f"{h - 1}:45" if h > 1 else "12:45"
    result = _re.sub(r'quarter\s+to\s+(\w+(?:-\w+)?)', _quarter_to, result)

    # Handle compound spoken minutes: "three thirty" -> "3:30", "six forty-five" -> "6:45"
    # Match <hour-word> <minute-word> (but NOT if minute-word is am/pm/o'clock)
    hour_words = [
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve'
    ]
    minute_words = [
        'ten', 'fifteen', 'twenty', 'twenty-one', 'twenty one',
        'twenty-two', 'twenty two', 'twenty-three', 'twenty three',
        'twenty-four', 'twenty four', 'twenty-five', 'twenty five',
        'thirty', 'thirty-five', 'thirty five', 'forty', 'forty-five',
        'forty five', 'fifty', 'fifty-five', 'fifty five',
    ]
    for hw in hour_words:
        for mw in sorted(minute_words, key=len, reverse=True):
            pattern = _re.compile(r'\b' + _re.escape(hw) + r'\s+' + _re.escape(mw) + r'\b')
            if pattern.search(result):
                h_digit = word_to_num[hw]
                m_digit = word_to_num.get(mw, mw)
                result = pattern.sub(f"{h_digit}:{m_digit}", result)

    # Replace remaining standalone number words with digits
    # Sort by length descending so "twenty-one" is matched before "twenty"
    for word, digit in sorted(word_to_num.items(), key=lambda x: len(x[0]), reverse=True):
        result = _re.sub(r'\b' + _re.escape(word) + r'\b', digit, result)

    return result


def _parse_reminder_regex(
    text: str,
    user_id: str,
    priority_override: Optional[str] = None
) -> dict:
    """
    Regex-based parsing (fallback method).
    
    Extracts:
    - Title and description
    - Category (medication, appointment, meal, etc.)
    - Time/date information
    - Recurrence pattern
    - Priority level
    """
    from datetime import timedelta
    import re
    
    # Normalize spoken numbers BEFORE any matching
    text_lower = _normalize_spoken_numbers(text)
    
    # Extract category based on keywords
    category = "general"
    if any(word in text_lower for word in ["medicine", "medication", "pill", "tablet", "drug"]):
        category = "medication"
    elif any(word in text_lower for word in ["doctor", "appointment", "visit", "checkup"]):
        category = "appointment"
    elif any(word in text_lower for word in ["breakfast", "lunch", "dinner", "meal", "eat"]):
        category = "meal"
    elif any(word in text_lower for word in ["shower", "bath", "hygiene", "brush", "wash"]):
        category = "hygiene"
    elif any(word in text_lower for word in ["exercise", "walk", "activity"]):
        category = "activity"
    
    # Extract priority
    priority = priority_override or "medium"
    if any(word in text_lower for word in ["urgent", "critical", "important", "asap"]):
        priority = "high"
    elif any(word in text_lower for word in ["when you can", "sometime", "eventually"]):
        priority = "low"
    
    # Extract recurrence pattern
    recurrence = None
    if any(word in text_lower for word in ["every day", "daily", "everyday"]):
        recurrence = "daily"
    elif any(word in text_lower for word in ["every week", "weekly"]):
        recurrence = "weekly"
    elif any(word in text_lower for word in ["every month", "monthly"]):
        recurrence = "monthly"
    
    # Extract DATE information first
    days_offset = 0
    now = datetime.now()
    
    # Check for relative day references
    if "tomorrow" in text_lower:
        days_offset = 1
    elif "day after tomorrow" in text_lower:
        days_offset = 2
    elif "today" in text_lower:
        days_offset = 0
    elif "next week" in text_lower:
        days_offset = 7
    
    # Check for specific weekdays
    weekdays = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    for day_name, day_num in weekdays.items():
        if day_name in text_lower:
            current_weekday = now.weekday()
            days_until = (day_num - current_weekday) % 7
            if days_until == 0:  # If it's today, assume next week
                days_until = 7
            days_offset = days_until
            break
    
    # Extract TIME information
    hour = None
    minute = 0
    time_found = False
    
    # Look for time patterns like "6 PM", "6:00 PM", "6.00 PM", "18:00"
    time_patterns = [
        (r'(\d{1,2})\s*[:\.]\s*(\d{2})\s*(am|pm|a\.m\.|p\.m\.)', 3),  # 6:00 PM, 6.00 PM
        (r'(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)', 2),  # 6 PM, 6 pm
        (r'(\d{1,2})\s*[:\.]\s*(\d{2})\b', 2),  # 18:00 (24-hour format)
        (r'at\s+(\d{1,2})(?:\s*o\'?clock)?', 1),  # at 6 o'clock
    ]
    
    for pattern, group_count in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                if group_count == 3:  # HH:MM AM/PM
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    period = match.group(3).lower().replace('.', '')
                    if period in ['pm', 'pm']:
                        if hour < 12:
                            hour += 12
                    elif period in ['am', 'am']:
                        if hour == 12:
                            hour = 0
                    time_found = True
                elif group_count == 2 and match.group(2) in ['am', 'pm', 'a.m.', 'p.m.']:  # HH AM/PM
                    hour = int(match.group(1))
                    minute = 0
                    period = match.group(2).lower().replace('.', '')
                    if period in ['pm', 'pm']:
                        if hour < 12:
                            hour += 12
                    elif period in ['am', 'am']:
                        if hour == 12:
                            hour = 0
                    time_found = True
                elif group_count == 2:  # 24-hour format HH:MM
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    time_found = True
                else:  # Just hour with "at"
                    hour = int(match.group(1))
                    minute = 0
                    # Assume PM if hour < 8 (for afternoon/evening reminders)
                    if hour < 8:
                        hour += 12
                    time_found = True
                break
            except (ValueError, AttributeError, IndexError):
                pass
    
    # Special time keywords (only if no specific time found)
    if not time_found:
        if "noon" in text_lower or "lunch time" in text_lower:
            hour, minute = 12, 0
            time_found = True
        elif "morning" in text_lower:
            hour, minute = 8, 0
            time_found = True
        elif "evening" in text_lower:
            hour, minute = 18, 0
            time_found = True
        elif "night" in text_lower or "bedtime" in text_lower:
            hour, minute = 21, 0
            time_found = True
    
    # Build the scheduled_time
    if time_found and hour is not None:
        # Create time for base date + days_offset
        base_date = now + timedelta(days=days_offset)
        scheduled_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If no explicit date was mentioned and time has passed today, schedule for tomorrow
        if days_offset == 0 and scheduled_time < now:
            scheduled_time += timedelta(days=1)
    else:
        # Default: 1 hour from now
        scheduled_time = now + timedelta(hours=1)
        scheduled_time = scheduled_time.replace(second=0, microsecond=0)
    
    # Generate title (extract main action)
    title = text[:50]  # First 50 chars
    if category == "medication":
        title = "Take Medication"
    elif category == "appointment":
        title = "Appointment Reminder"
    elif category == "meal":
        title = "Meal Reminder"
    
    # Add time context to title if recurrence exists
    if recurrence:
        title = f"{title} ({recurrence.capitalize()})"
    
    return {
        "title": title,
        "description": text,
        "category": category,
        "priority": priority,
        "scheduled_time": scheduled_time,
        "recurrence": recurrence
    }


def get_db_service():
    """Get database service instance (lazy initialization)."""
    global _db_service
    if _db_service is None:
        _db_service = ReminderDatabaseService()
    return _db_service


async def _save_caregiver_alert(
    *,
    patient_id: str,
    caregiver_ids: list,
    alert_type: str,          # "missed_medication" | "snoozed_reminder" | "new_high_priority"
    priority: str,            # "high" | "critical"
    reminder_id: str,
    reminder_title: str,
    reminder_category: str,
    scheduled_time: Optional[datetime] = None,
):
    """
    Persist a caregiver alert document to the caregiver_alerts collection.
    Called whenever a medication reminder is missed, snoozed, or created at
    high / critical priority so the caregiver dashboard can surface it.
    """
    try:
        if not caregiver_ids:
            return   # nothing to notify

        type_labels = {
            "missed_medication":    f"Missed medication: {reminder_title}",
            "snoozed_reminder":     f"Reminder snoozed: {reminder_title}",
            "new_high_priority":    f"New {priority} priority reminder: {reminder_title}",
        }
        type_messages = {
            "missed_medication":  (
                f"{reminder_category.capitalize()} reminder '{reminder_title}' was MISSED"
                + (f" (scheduled {scheduled_time.strftime('%H:%M')})" if scheduled_time else "") + "."
            ),
            "snoozed_reminder": (
                f"{reminder_category.capitalize()} reminder '{reminder_title}' was snoozed"
                + (f" (was due {scheduled_time.strftime('%H:%M')})" if scheduled_time else "") + "."
            ),
            "new_high_priority": (
                f"A {priority}-priority {reminder_category} reminder '{reminder_title}' was created"
                + (f" for {scheduled_time.strftime('%H:%M')}." if scheduled_time else ".")
            ),
        }

        alert_doc = {
            "patient_id":        patient_id,
            "caregiver_ids":     caregiver_ids,
            "title":             type_labels.get(alert_type, f"Reminder alert: {reminder_title}"),
            "message":           type_messages.get(alert_type, ""),
            "priority":          priority,
            "type":              alert_type,
            "reminder_id":       reminder_id,
            "reminder_title":    reminder_title,
            "reminder_category": reminder_category,
            "created_at":        datetime.now(),
            "resolved":          False,
        }

        alerts_col = Database.get_collection("caregiver_alerts")
        await alerts_col.insert_one(alert_doc)
        logger.info(
            f"Caregiver alert saved: type={alert_type}, patient={patient_id}, "
            f"reminder={reminder_id}, priority={priority}"
        )
    except Exception as e:
        logger.warning(f"Failed to save caregiver alert: {e}")


@router.post("/create", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_reminder(reminder: Reminder):
    """
    Create a new reminder for a user.
    
    - **user_id**: User identifier
    - **title**: Reminder title
    - **description**: Optional detailed description
    - **scheduled_time**: When to send reminder
    - **priority**: Priority level (low, medium, high, critical)
    - **category**: Reminder category (medication, appointment, meal, etc.)
    - **caregiver_ids**: List of caregiver IDs to notify
    """
    try:
        # Auto-generate ID if not provided
        if not reminder.id:
            reminder.id = f"reminder_{uuid.uuid4().hex[:12]}"
        
        # Set creation timestamps
        reminder.created_at = datetime.now()
        reminder.updated_at = datetime.now()
        
        # Save to MongoDB database
        db_service = get_db_service()
        db_result = await db_service.create_reminder(reminder)
        
        logger.info(f"Created reminder {reminder.id} for user {reminder.user_id}")

        # ── Auto-alert caregivers for high/critical medication reminders ─────
        if (
            reminder.caregiver_ids
            and reminder.category == "medication"
            and reminder.priority in (ReminderPriority.HIGH, ReminderPriority.CRITICAL)
        ):
            await _save_caregiver_alert(
                patient_id=reminder.user_id,
                caregiver_ids=reminder.caregiver_ids,
                alert_type="new_high_priority",
                priority=reminder.priority.value,
                reminder_id=reminder.id,
                reminder_title=reminder.title,
                reminder_category=reminder.category,
                scheduled_time=reminder.scheduled_time,
            )

        return {
            "status": "success",
            "message": "Reminder created successfully",
            "reminder": reminder.dict(),
            "database_id": db_result.get("id")
        }
        
    except Exception as e:
        logger.error(f"Error creating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/create-from-audio", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_reminder_from_audio(
    user_id: str = Form(..., description="User identifier"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, flac)"),
    priority: Optional[str] = Form("medium", description="Reminder priority (low, medium, high, critical)"),
    caregiver_ids: Optional[str] = Form(None, description="Comma-separated caregiver IDs"),
    user_timezone: Optional[str] = Form(None, description="IANA timezone name e.g. Asia/Colombo")
):
    """
    Create reminder from audio recording using Whisper transcription + NLP parsing.
    
    **Flow:**
    1. Upload audio file
    2. Transcribe using Whisper (local, free)
    3. Parse reminder details from transcription (NLP)
    4. Create reminder automatically
    
    **Example audio commands:**
    - "Remind me to take my blood pressure medicine at 8 AM every morning"
    - "Set a reminder for doctor appointment next Tuesday at 2 PM"
    - "I need to remember my lunch at noon daily"
    
    **Parameters:**
    - **user_id**: User identifier
    - **file**: Audio file containing the reminder command
    - **priority**: Optional priority override (default: extracted from audio)
    - **caregiver_ids**: Optional caregiver IDs to notify (comma-separated)
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        logger.info(f"Creating reminder from audio for user: {user_id}")
        
        # Get file extension
        file_ext = Path(file.filename).suffix or ".wav"
        
        # Log file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        logger.info(f"Processing audio file... Size: {file_size} bytes")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        try:
            # Step 1: Transcribe audio using Whisper
            logger.info("📝 Transcribing audio with Whisper...")
            whisper_service = get_whisper_service()
            transcription_result = whisper_service.transcribe(
                audio_path=temp_audio_path,
                language=None  # Auto-detect
            )
            
            transcribed_text = transcription_result["text"]
            logger.info(f"✅ Transcription: '{transcribed_text}'")

            # Reject empty or hallucinated-noise-only outputs from Whisper
            _stripped = transcribed_text.strip()
            _is_noise = not _stripped or all(c in ". \t\n" for c in _stripped)
            if _is_noise:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Could not transcribe audio. "
                        "Please speak clearly and ensure your microphone is working. "
                        "Try recording in a quieter environment."
                    )
                )
            
            # Step 2: Parse reminder details using NLP
            logger.info("🧠 Parsing reminder details from transcription...")
            reminder_details = _parse_reminder_from_text(
                text=transcribed_text,
                user_id=user_id,
                priority_override=priority
            )
            
            # Convert extracted naive local time to UTC using user's timezone
            if user_timezone:
                try:
                    user_tz = ZoneInfo(user_timezone)
                    naive_local = reminder_details["scheduled_time"]
                    local_aware = naive_local.replace(tzinfo=user_tz)
                    utc_dt = local_aware.astimezone(timezone.utc).replace(tzinfo=None)
                    reminder_details["scheduled_time"] = utc_dt
                    logger.info(f"🌍 Converted {naive_local} ({user_timezone}) → {utc_dt} UTC")
                except Exception as tz_err:
                    logger.warning(f"Timezone conversion failed ({user_timezone}): {tz_err}")

            # Step 3: Create reminder
            reminder_id = f"reminder_{uuid.uuid4().hex[:12]}"
            
            # Parse caregiver IDs if provided
            caregiver_list = []
            if caregiver_ids:
                caregiver_list = [cid.strip() for cid in caregiver_ids.split(",") if cid.strip()]
            
            reminder = Reminder(
                id=reminder_id,
                user_id=user_id,
                title=reminder_details["title"],
                description=reminder_details["description"],
                scheduled_time=reminder_details["scheduled_time"],
                priority=ReminderPriority(reminder_details["priority"]),
                category=reminder_details["category"],
                recurrence=reminder_details.get("recurrence"),
                caregiver_ids=caregiver_list,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Save to database
            db_service = get_db_service()
            db_result = await db_service.create_reminder(reminder)
            
            logger.info(f"✅ Reminder created from audio: {reminder_id}")
            
            return {
                "status": "success",
                "message": "Reminder created successfully from audio",
                "reminder": reminder.dict(),
                "transcription": transcribed_text,
                "audio_file": file.filename,
                "db_status": db_result
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating reminder from audio: {e}", exc_info=True)
        
        # Provide helpful error messages based on the error type
        error_msg = str(e).lower()
        
        if "format not recognised" in error_msg or ".3gp" in error_msg or "cannot load audio" in error_msg:
            detail = (
                "Your device recorded audio in .3gp format, which requires FFmpeg to process. "
                "SOLUTION: Please install FFmpeg from https://ffmpeg.org/download.html\n\n"
                "For Windows:\n"
                "1. Download from: https://ffmpeg.org/download.html#build-windows\n"
                "2. Extract the files to a folder (e.g., C:\\ffmpeg)\n"
                "3. Add the folder to your system PATH environment variable\n"
                "4. Restart your application\n\n"
                "After installing FFmpeg, audio recording will work automatically."
            )
        elif "no backend" in error_msg or "nobackenderror" in error_msg:
            detail = (
                "Audio processing failed - FFmpeg is required. "
                "Please install FFmpeg from: https://ffmpeg.org/download.html"
            )
        else:
            detail = f"Failed to process audio: {str(e)[:200]}"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


@router.post("/natural-language", response_model=dict)
async def create_reminder_from_natural_language(command: ReminderCommand):
    """
    Create reminder from natural language text command using NLP.
    
    Examples:
    - "Remind me to take my tablets after lunch"
    - "Set medication reminder for 8 AM daily"
    - "Doctor appointment next Tuesday at 2 PM"
    
    - **user_id**: User identifier
    - **command_text**: Natural language command
    - **audio_path**: Optional audio recording path
    """
    try:
        logger.info(f"Parsing natural language command for user {command.user_id}")
        
        # Parse natural language command
        parsed_reminder = _parse_reminder_from_text(
            text=command.command_text,
            user_id=command.user_id
        )
        
        logger.info(f"Parsed NL command for user {command.user_id}")
        
        return {
            "status": "success",
            "message": "Reminder parsed from natural language",
            "parsed_reminder": parsed_reminder,
            "original_command": command.command_text
        }
        
    except Exception as e:
        logger.error(f"Error parsing NL command: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/respond", response_model=dict)
async def process_reminder_response(response: ReminderResponse):
    """
    Process user's response to a reminder notification.
    
    Analyzes response using Pitt Corpus trained models to detect:
    - Confusion or memory issues
    - Confirmation or completion
    - Need for caregiver intervention
    
    - **reminder_id**: Reminder identifier
    - **user_id**: User identifier
    - **response_text**: User's text response
    - **audio_path**: Optional audio recording
    """
    try:
        # Get reminder details (TODO: from database)
        # For now, create mock reminder
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        reminder = Reminder(
            id=response.reminder_id,
            user_id=response.user_id,
            title="Sample Reminder",
            scheduled_time=datetime.now(),
            priority=ReminderPriority.MEDIUM,
            category="medication"
        )
        
        # Process response through adaptive scheduler
        result = scheduler.process_reminder_response(
            reminder=reminder,
            user_response=response.response_text,
            audio_path=response.audio_path
        )
        
        return {
            "status": "success",
            "message": "Response processed successfully",
            "analysis": result['analysis'],
            "interaction_type": result['analysis']['interaction_type'],
            "recommended_action": result['analysis']['recommended_action'],
            "caregiver_notified": result.get('action_result', {}).get('caregiver_notified', False),
            "cognitive_risk_score": result['analysis']['cognitive_risk_score']
        }
        
    except Exception as e:
        logger.error(f"Error processing response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/user/{user_id}", response_model=dict)
async def get_user_reminders(
    user_id: str,
    status_filter: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get all reminders for a user.
    
    - **user_id**: User identifier
    - **status_filter**: Optional filter by status (active, completed, missed, etc.)
    - **category**: Optional filter by category
    """
    try:
        # Convert status_filter string to enum if provided
        status_enum = None
        if status_filter:
            try:
                status_enum = ReminderStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}. Valid values: active, completed, missed, snoozed, cancelled"
                )
        
        # Query database for user reminders
        try:
            db_service = get_db_service()
            reminders = await db_service.get_user_reminders(user_id, status_enum)
        except RuntimeError as db_error:
            logger.error(f"Database not connected: {db_error}")
            # Return empty result if DB not connected
            reminders = []
        
        # Apply category filter if specified
        if category:
            reminders = [r for r in reminders if r.get("category") == category]
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_reminders": len(reminders),
            "reminders": reminders
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reminders: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reminders: {str(e)}"
        )


@router.get("/user/{user_id}/due-now", response_model=dict)
async def get_due_reminders_now(user_id: str, time_window_minutes: int = 5):
    """
    Get reminders that are due NOW for a user (for polling-based notifications).
    
    Mobile apps can call this endpoint every 30-60 seconds to check for due reminders
    and trigger local alarms/notifications.
    
    - **user_id**: User identifier
    - **time_window_minutes**: Look ahead window in minutes (default: 5)
    
    Returns reminders that should trigger an alarm/notification right now.
    """
    try:
        db_service = get_db_service()
        
        # Get reminders due within the time window (filtered by user_id in DB)
        try:
            due_reminders = await db_service.get_due_reminders(
                time_window_minutes=time_window_minutes,
                user_id=user_id
            )
        except RuntimeError as db_error:
            logger.error(f"Database not connected: {db_error}")
            return {
                "status": "error",
                "message": "Database not connected",
                "user_id": user_id,
                "due_reminders": [],
                "should_trigger_alarm": False
            }

        # Already filtered by user_id at the DB level
        user_due_reminders = due_reminders
        
        # Check if any are due RIGHT NOW (within the lookback window or upcoming <=1 min)
        # Use datetime.now() — matches how scheduled_time is stored (naive local time)
        now = datetime.now()
        urgent_reminders = []
        for reminder in user_due_reminders:
            scheduled_time = reminder.get("scheduled_time")

            # Handle both string and datetime objects
            if isinstance(scheduled_time, str):
                try:
                    scheduled_time = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
                    # Strip tz for naive-local comparison
                    if scheduled_time.tzinfo is not None:
                        from datetime import timezone
                        scheduled_time = scheduled_time.astimezone(timezone.utc).replace(tzinfo=None)
                except Exception:
                    continue  # Skip if can't parse
            elif not isinstance(scheduled_time, datetime):
                continue  # Skip if not string or datetime

            time_diff = (scheduled_time - now).total_seconds() / 60.0  # in minutes

            # Trigger alarm for:
            #   - Already past (up to 10 min ago) — catches reminders missed during gaps/restarts
            #   - About to fire (within next 1 min) — early alert
            if -10 <= time_diff <= 1:
                # Convert datetime to ISO string for JSON response
                reminder_copy = reminder.copy()
                if isinstance(reminder_copy.get("scheduled_time"), datetime):
                    reminder_copy["scheduled_time"] = reminder_copy["scheduled_time"].isoformat()
                
                urgent_reminders.append({
                    **reminder_copy,
                    "time_until_due_minutes": round(time_diff, 1),
                    "trigger_alarm": True
                })
        
        # Mark urgent reminders as 'awaiting_acknowledgment' so subsequent
        # polls do not keep returning (and re-triggering) the same alarm.
        for ur in urgent_reminders:
            ur_id = ur.get("id")
            if ur_id:
                try:
                    await db_service.update_reminder(
                        ur_id,
                        {
                            "status": "awaiting_acknowledgment",
                            "delivered_at": now.isoformat(),
                            "alarm_triggered_at": now.isoformat(),
                        }
                    )
                except Exception as upd_err:
                    logger.warning(f"Failed to mark reminder {ur_id} as awaiting_acknowledgment: {upd_err}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "current_time": now.isoformat(),
            "total_due_soon": len(user_due_reminders),
            "urgent_count": len(urgent_reminders),
            "should_trigger_alarm": len(urgent_reminders) > 0,
            "urgent_reminders": urgent_reminders,
            "upcoming_reminders": user_due_reminders[:5]  # Next 5 upcoming
        }
        
    except Exception as e:
        logger.error(f"Error getting due reminders: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/behavior/{user_id}", response_model=dict)
async def get_user_behavior_pattern(
    user_id: str,
    reminder_id: Optional[str] = None,
    category: Optional[str] = None,
    days: int = 30
):
    """
    Get user's behavior patterns and analytics.
    
    Returns statistics on:
    - Confirmation rates
    - Confusion frequency
    - Optimal reminder times
    - Cognitive risk trends
    
    - **user_id**: User identifier
    - **reminder_id**: Optional specific reminder
    - **category**: Optional reminder category
    - **days**: Analysis period (default 30 days)
    """
    try:
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            reminder_id=reminder_id,
            category=category,
            days=days
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "pattern": pattern.dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting behavior pattern: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/schedule/{reminder_id}", response_model=dict)
async def get_optimal_schedule(reminder_id: str):
    """
    Get optimal schedule recommendation for a reminder.

    Returns adaptive scheduling recommendations based on:
    - Historical user behavior
    - Optimal response times
    - Cognitive risk patterns

    - **reminder_id**: Reminder identifier
    """
    try:
        db_service = get_db_service()
        reminder_data = await db_service.get_reminder(reminder_id)

        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )

        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority

        scheduled_time = reminder_data.get("scheduled_time")
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace("+00:00", ""))
        elif not isinstance(scheduled_time, datetime):
            scheduled_time = datetime.now()

        reminder = Reminder(
            id=reminder_id,
            user_id=reminder_data["user_id"],
            title=reminder_data.get("title", "Reminder"),
            description=reminder_data.get("description"),
            scheduled_time=scheduled_time,
            priority=ReminderPriority(reminder_data.get("priority", "medium")),
            category=reminder_data.get("category", "general"),
        )

        schedule_info = scheduler.get_optimal_reminder_schedule(reminder)

        # Convert datetime to ISO string for JSON serialization
        if isinstance(schedule_info.get("optimal_time"), datetime):
            schedule_info["optimal_time"] = schedule_info["optimal_time"].isoformat()

        return {
            "status": "success",
            "reminder_id": reminder_id,
            "user_id": reminder_data["user_id"],
            "schedule": schedule_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schedule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/update/{reminder_id}", response_model=dict)
async def update_reminder(reminder_id: str, updated_reminder: Reminder):
    """
    Update an existing reminder.
    
    - **reminder_id**: Reminder identifier
    - **updated_reminder**: Updated reminder data
    """
    try:
        updated_reminder.id = reminder_id
        updated_reminder.updated_at = datetime.now()
        
        # Build update payload — preserve created_at, replace everything else
        update_data = updated_reminder.dict()
        update_data.pop("id", None)        # _id is the query key, not a $set field
        update_data.pop("created_at", None)  # Do not overwrite original creation time
        update_data["updated_at"] = datetime.now()
        
        db_service = get_db_service()
        db_result = await db_service.update_reminder(reminder_id, update_data)
        
        if db_result.get("error") == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        logger.info(f"Updated reminder {reminder_id}")
        
        return {
            "status": "success",
            "message": "Reminder updated successfully",
            "reminder": updated_reminder.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/delete/{reminder_id}", response_model=dict)
async def delete_reminder(reminder_id: str):
    """
    Delete a reminder.
    
    - **reminder_id**: Reminder identifier
    """
    try:
        db_service = get_db_service()
        deleted = await db_service.delete_reminder(reminder_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        logger.info(f"Deleted reminder {reminder_id}")
        
        return {
            "status": "success",
            "message": "Reminder deleted successfully",
            "reminder_id": reminder_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/snooze/{reminder_id}", response_model=dict)
async def snooze_reminder(reminder_id: str, delay_minutes: int = 15, user_id: Optional[str] = None):
    """
    Snooze a reminder for specified duration.
    
    - **reminder_id**: Reminder identifier
    - **delay_minutes**: Minutes to delay (default 15)
    - **user_id**: User performing the snooze (for behavior tracking)
    """
    try:
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority

        db_service = get_db_service()
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )

        effective_user = user_id or reminder_data.get("user_id", "unknown")
        scheduled_time = reminder_data.get("scheduled_time")
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace("+00:00", ""))

        reminder_obj = Reminder(
            id=reminder_id,
            user_id=effective_user,
            title=reminder_data.get("title", "Reminder"),
            scheduled_time=scheduled_time or datetime.now(),
            priority=ReminderPriority(reminder_data.get("priority", "medium")),
            category=reminder_data.get("category", "general"),
            caregiver_ids=reminder_data.get("caregiver_ids", []),
            description=reminder_data.get("description"),
        )

        updated_reminder = scheduler.reschedule_reminder(
            reminder=reminder_obj,
            delay_minutes=delay_minutes,
            reason="user_requested_snooze"
        )

        new_time = updated_reminder.scheduled_time
        await db_service.update_reminder(
            reminder_id,
            {
                "status": "snoozed",
                "scheduled_time": new_time.isoformat(),
                "snoozed_at": datetime.now().isoformat(),
                "snooze_delay_minutes": delay_minutes,
                "snooze_count": reminder_data.get("snooze_count", 0) + 1,
            }
        )

        # Log interaction for behavior tracking
        try:
            snooze_interaction = ReminderInteraction(
                reminder_id=reminder_id,
                user_id=effective_user,
                reminder_category=reminder_obj.category,
                interaction_type=InteractionType.DELAYED,
                interaction_time=datetime.now(),
                response_time_seconds=None,
                user_response_text=f"Snoozed for {delay_minutes} minutes"
            )
            behavior_tracker.log_interaction(snooze_interaction)
        except Exception as bt_err:
            logger.warning(f"Behavior tracking failed on snooze: {bt_err}")

        # ── Alert caregivers for snoozed medication reminders ──────────────
        if (
            reminder_obj.caregiver_ids
            and reminder_obj.category == "medication"
            and reminder_obj.priority in (ReminderPriority.HIGH, ReminderPriority.CRITICAL)
        ):
            await _save_caregiver_alert(
                patient_id=effective_user,
                caregiver_ids=reminder_obj.caregiver_ids,
                alert_type="snoozed_reminder",
                priority=reminder_obj.priority.value,
                reminder_id=reminder_id,
                reminder_title=reminder_obj.title,
                reminder_category=reminder_obj.category,
                scheduled_time=scheduled_time,
            )

        return {
            "status": "success",
            "message": f"Reminder snoozed for {delay_minutes} minutes",
            "new_scheduled_time": new_time.isoformat(),
            "snooze_count": reminder_data.get("snooze_count", 0) + 1,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error snoozing reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Caregiver endpoints

@router.get("/caregiver/alerts/{caregiver_id}", response_model=dict)
async def get_caregiver_alerts(caregiver_id: str, active_only: bool = True):
    """
    Get alerts for a caregiver.
    
    - **caregiver_id**: Caregiver identifier
    - **active_only**: Only return unresolved alerts (default True)
    """
    try:
        if active_only:
            alerts = caregiver_notifier.get_active_alerts(caregiver_id)
        else:
            # TODO: Get all alerts from database
            alerts = []
        
        return {
            "status": "success",
            "caregiver_id": caregiver_id,
            "total_alerts": len(alerts),
            "alerts": [alert.dict() for alert in alerts]
        }
        
    except Exception as e:
        logger.error(f"Error getting caregiver alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/caregiver/alerts/{alert_id}/acknowledge", response_model=dict)
async def acknowledge_alert(alert_id: str, caregiver_id: str):
    """
    Acknowledge a caregiver alert.
    
    - **alert_id**: Alert identifier
    - **caregiver_id**: Caregiver acknowledging the alert
    """
    try:
        success = caregiver_notifier.acknowledge_alert(alert_id, caregiver_id)
        
        if success:
            return {
                "status": "success",
                "message": "Alert acknowledged",
                "alert_id": alert_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to acknowledge alert"
            )
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/caregiver/alerts/{alert_id}/resolve", response_model=dict)
async def resolve_alert(alert_id: str, caregiver_id: str):
    """
    Mark an alert as resolved.
    
    - **alert_id**: Alert identifier
    - **caregiver_id**: Caregiver resolving the alert
    """
    try:
        success = caregiver_notifier.resolve_alert(alert_id, caregiver_id)
        
        if success:
            return {
                "status": "success",
                "message": "Alert resolved",
                "alert_id": alert_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to resolve alert"
            )
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/analytics/dashboard/{user_id}", response_model=dict)
async def get_user_dashboard(user_id: str, days: int = 7):
    """
    Get comprehensive dashboard analytics for a user.
    
    Returns:
    - Reminder statistics
    - Cognitive risk trends
    - Behavior patterns
    - Recent interactions
    
    - **user_id**: User identifier
    - **days**: Analysis period (default 7 days)
    """
    try:
        # Get behavior pattern
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            days=days
        )
        
        # Calculate statistics
        total = pattern.total_reminders
        confirmation_rate = (pattern.confirmed_count / total) if total > 0 else 0
        confusion_rate = (pattern.confused_count / total) if total > 0 else 0
        
        dashboard = {
            "user_id": user_id,
            "period_days": days,
            "statistics": {
                "total_reminders": total,
                "confirmed": pattern.confirmed_count,
                "ignored": pattern.ignored_count,
                "delayed": pattern.delayed_count,
                "confused": pattern.confused_count,
                "confirmation_rate": confirmation_rate,
                "confusion_rate": confusion_rate
            },
            "cognitive_health": {
                "avg_risk_score": pattern.avg_cognitive_risk_score,
                "trend": pattern.confusion_trend,
                "escalation_recommended": pattern.escalation_recommended
            },
            "timing": {
                "optimal_hour": pattern.optimal_reminder_hour,
                "worst_hours": pattern.worst_response_hours,
                "avg_response_time_seconds": pattern.avg_response_time_seconds
            },
            "recommendations": {
                "frequency_multiplier": pattern.recommended_frequency_multiplier,
                "time_adjustment_minutes": pattern.recommended_time_adjustment_minutes
            }
        }
        
        return {
            "status": "success",
            "dashboard": dashboard
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/score/{user_id}", response_model=dict)
async def get_user_score(
    user_id: str,
    days: int = 30,
    recent_responses: Optional[str] = None,
):
    """
    Get composite cognitive score for a user (0-100%).

    Combines ML model predictions with reminder adherence data:
    - **Cognitive risk** (40%): Dementia-risk model output
    - **Confusion & severity** (20%): Confusion detection + severity classifier
    - **Adherence** (25%): Confirmed vs missed/ignored reminders
    - **Behavioral trend** (15%): Improving or declining patterns

    Lower score = healthier. Higher score = more concern.

    - **user_id**: User identifier
    - **days**: Analysis period in days (default 30)
    - **recent_responses**: Pipe-separated recent text responses to run through
      ML models (e.g. "I took my medicine|What was I doing?|Yes I remember")
    """
    try:
        responses = []
        if recent_responses:
            responses = [r.strip() for r in recent_responses.split("|") if r.strip()]

        result = cognitive_score_service.compute_user_score(
            user_id=user_id,
            recent_responses=responses,
            days=days,
        )
        return {"status": "success", "score": result}

    except Exception as e:
        logger.error(f"Error computing user score: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/reports/weekly/{user_id}", response_model=dict)
async def get_weekly_report(
    user_id: str,
    end_date: Optional[str] = None,
    format: str = "json"
):
    """
    Generate comprehensive weekly risk monitoring report.
    
    Returns detailed analysis including:
    - Cognitive risk trends (daily and weekly)
    - Reminder completion statistics
    - Confusion patterns and memory issues
    - Caregiver alert frequency
    - Time-of-day performance analysis
    - Category-specific breakdowns
    - Actionable recommendations
    - Week-over-week comparison
    
    - **user_id**: Patient identifier
    - **end_date**: End date for report (ISO 8601 format, default: today)
    - **format**: Output format ('json' or 'pdf', default: 'json')
    """
    try:
        # Parse end_date if provided
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end_date format. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
                )
        else:
            end_dt = datetime.now()

        start_dt = end_dt - timedelta(days=7)

        # ----- Query reminders from MongoDB directly -----
        from src.database import Database
        reminders_col = Database.get_collection("reminders")

        query = {"user_id": user_id}
        cursor = reminders_col.find(query)
        all_reminders = await cursor.to_list(length=5000)

        # Partition into current week and previous week
        week_reminders = []
        prev_week_reminders = []
        prev_start = start_dt - timedelta(days=7)

        for r in all_reminders:
            created = r.get("created_at") or r.get("scheduled_time")
            if created is None:
                continue
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created)
                except (ValueError, TypeError):
                    continue
            if start_dt <= created <= end_dt:
                week_reminders.append(r)
            elif prev_start <= created < start_dt:
                prev_week_reminders.append(r)

        total = len(week_reminders)

        # Status counts
        completed = sum(1 for r in week_reminders if r.get("status") == "completed")
        missed = sum(1 for r in week_reminders if r.get("status") == "missed")
        active = sum(1 for r in week_reminders if r.get("status") in ("active", "snoozed"))
        acknowledged = sum(1 for r in week_reminders if r.get("status") == "acknowledged")
        completion_rate = (completed + acknowledged) / total if total > 0 else 0

        # Cognitive risk scores
        risk_scores = [
            r["cognitive_risk_score"]
            for r in week_reminders
            if r.get("cognitive_risk_score") is not None
        ]
        avg_risk = statistics.mean(risk_scores) if risk_scores else 0.0
        peak_risk = max(risk_scores) if risk_scores else 0.0
        lowest_risk = min(risk_scores) if risk_scores else 0.0

        # Response times
        response_times = [
            r["response_time_seconds"]
            for r in week_reminders
            if r.get("response_time_seconds") is not None
        ]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0

        # Daily summaries
        from collections import defaultdict as _defaultdict
        daily_buckets = _defaultdict(list)
        for r in week_reminders:
            created = r.get("created_at") or r.get("scheduled_time")
            if created is None:
                continue
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created)
                except (ValueError, TypeError):
                    continue
            daily_buckets[created.date().isoformat()].append(r)

        daily_summaries = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_key = current_date.isoformat()
            day_items = daily_buckets.get(date_key, [])
            day_risk = [
                x["cognitive_risk_score"]
                for x in day_items
                if x.get("cognitive_risk_score") is not None
            ]
            day_completed = sum(
                1 for x in day_items if x.get("status") in ("completed", "acknowledged")
            )
            daily_summaries.append({
                "date": date_key,
                "avg_cognitive_risk": statistics.mean(day_risk) if day_risk else 0.0,
                "confusion_count": sum(
                    1 for x in day_items
                    if x.get("user_response", "").lower().find("confus") >= 0
                ),
                "total_interactions": len(day_items),
                "completion_rate": day_completed / len(day_items) if day_items else 0.0,
                "alert_count": sum(
                    1 for x in day_items if x.get("notify_caregiver_on_miss") and x.get("status") == "missed"
                ),
            })
            current_date += timedelta(days=1)

        # Hour-of-day analysis
        hour_performance = _defaultdict(lambda: {"total": 0, "completed": 0})
        for r in week_reminders:
            sched = r.get("scheduled_time")
            if sched is None:
                continue
            if isinstance(sched, str):
                try:
                    sched = datetime.fromisoformat(sched)
                except (ValueError, TypeError):
                    continue
            h = sched.hour
            hour_performance[h]["total"] += 1
            if r.get("status") in ("completed", "acknowledged"):
                hour_performance[h]["completed"] += 1

        sorted_hours = sorted(
            hour_performance.items(),
            key=lambda x: x[1]["completed"] / x[1]["total"] if x[1]["total"] else 0,
            reverse=True,
        )
        best_hours = [h for h, _ in sorted_hours[:3]] if sorted_hours else []
        worst_hours = [h for h, _ in sorted_hours[-3:]] if sorted_hours else []

        # Category breakdown
        cat_buckets = _defaultdict(lambda: {"total": 0, "completed": 0, "risk": []})
        for r in week_reminders:
            cat = r.get("category", "general")
            cat_buckets[cat]["total"] += 1
            if r.get("status") in ("completed", "acknowledged"):
                cat_buckets[cat]["completed"] += 1
            if r.get("cognitive_risk_score") is not None:
                cat_buckets[cat]["risk"].append(r["cognitive_risk_score"])

        category_breakdown = {}
        for cat, data in cat_buckets.items():
            category_breakdown[cat] = {
                "total": data["total"],
                "completed": data["completed"],
                "completion_rate": data["completed"] / data["total"] if data["total"] else 0,
                "avg_risk": statistics.mean(data["risk"]) if data["risk"] else 0.0,
            }

        # Caregiver alerts from alerts collection
        alerts_col = Database.get_collection("caregiver_alerts")
        alert_query = {"user_id": user_id, "created_at": {"$gte": start_dt, "$lte": end_dt}}
        alert_cursor = alerts_col.find(alert_query)
        alerts = await alert_cursor.to_list(length=1000)
        total_alerts = len(alerts)
        critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical")
        high_alerts = sum(1 for a in alerts if a.get("severity") == "high")
        unresolved_alerts = sum(1 for a in alerts if not a.get("is_resolved", False))

        # Risk trend
        active_days = [d for d in daily_summaries if d["total_interactions"] > 0]
        if len(active_days) >= 3:
            scores = [d["avg_cognitive_risk"] for d in active_days]
            first_half = statistics.mean(scores[: len(scores) // 2])
            second_half = statistics.mean(scores[len(scores) // 2 :])
            if second_half < first_half - 0.05:
                risk_trend = "improving"
            elif second_half > first_half + 0.05:
                risk_trend = "declining"
            else:
                risk_trend = "stable"
        else:
            risk_trend = "stable" if active_days else "insufficient_data"

        # Previous week comparison
        prev_risk = [
            r["cognitive_risk_score"]
            for r in prev_week_reminders
            if r.get("cognitive_risk_score") is not None
        ]
        prev_avg_risk = statistics.mean(prev_risk) if prev_risk else None
        risk_change = None
        if prev_avg_risk is not None and prev_avg_risk > 0:
            risk_change = round(((avg_risk - prev_avg_risk) / prev_avg_risk) * 100, 1)

        # Risk level
        if avg_risk >= 0.75:
            risk_level = "critical"
        elif avg_risk >= 0.6:
            risk_level = "high"
        elif avg_risk >= 0.4:
            risk_level = "moderate"
        elif total > 0:
            risk_level = "low"
        else:
            risk_level = "unknown"

        # Recommendations
        recommendations = []
        if completion_rate < 0.6 and total > 0:
            recommendations.append(
                "Low completion rate detected. Consider adjusting reminder frequency and timing."
            )
        if avg_risk > 0.7:
            recommendations.append(
                "High average cognitive risk. Recommend immediate medical evaluation."
            )
        elif avg_risk > 0.5:
            recommendations.append(
                "Elevated cognitive risk. Schedule follow-up with healthcare provider."
            )
        if risk_trend == "declining":
            recommendations.append(
                "Cognitive performance declining. Increase caregiver monitoring."
            )
        elif risk_trend == "improving":
            recommendations.append(
                "Cognitive performance improving. Continue current care plan."
            )
        if best_hours:
            recommendations.append(
                f"Best response hours: {', '.join(f'{h}:00' for h in best_hours)}. "
                "Schedule important reminders during these times."
            )
        if unresolved_alerts > 3:
            recommendations.append(
                f"{unresolved_alerts} unresolved alerts require caregiver follow-up."
            )
        if not recommendations:
            recommendations.append("Patient performance is stable. Continue current monitoring.")

        intervention_needed = (
            avg_risk > 0.6
            or (risk_trend == "declining" and avg_risk > 0.4)
            or (completion_rate < 0.5 and total > 0)
            or unresolved_alerts > 2
        )
        escalation_required = avg_risk >= 0.75 or critical_alerts >= 3 or peak_risk >= 0.85

        confusion_count = sum(1 for d in daily_summaries for _ in range(d["confusion_count"]))

        report = {
            "user_id": user_id,
            "report_period_start": start_dt.isoformat(),
            "report_period_end": end_dt.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "total_reminders": total,
            "completed_reminders": completed + acknowledged,
            "missed_reminders": missed,
            "active_reminders": active,
            "completion_rate": round(completion_rate, 3),
            "avg_cognitive_risk_score": round(avg_risk, 3),
            "peak_cognitive_risk_score": round(peak_risk, 3),
            "lowest_cognitive_risk_score": round(lowest_risk, 3),
            "risk_trend": risk_trend,
            "confusion_count": confusion_count,
            "confirmed_count": completed + acknowledged,
            "ignored_count": missed,
            "delayed_count": sum(1 for r in week_reminders if r.get("status") == "snoozed"),
            "daily_summaries": daily_summaries,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "high_priority_alerts": high_alerts,
            "unresolved_alerts": unresolved_alerts,
            "best_response_hours": best_hours,
            "worst_response_hours": worst_hours,
            "avg_response_time_seconds": round(avg_response_time, 1),
            "category_breakdown": category_breakdown,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "intervention_needed": intervention_needed,
            "escalation_required": escalation_required,
            "previous_week_avg_risk": round(prev_avg_risk, 3) if prev_avg_risk is not None else None,
            "risk_change_percentage": risk_change,
        }

        return {"status": "success", "report": report}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/reports/weekly/{user_id}/summary", response_model=dict)
async def get_weekly_report_summary(user_id: str):
    """
    Get brief summary of weekly report (for dashboard display).
    
    Returns key metrics only:
    - Risk level and score
    - Completion rate
    - Alert count
    - Intervention status
    
    - **user_id**: Patient identifier
    """
    try:
        from src.database import Database

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)

        reminders_col = Database.get_collection("reminders")
        cursor = reminders_col.find({
            "user_id": user_id,
            "created_at": {"$gte": start_dt, "$lte": end_dt},
        })
        week_reminders = await cursor.to_list(length=5000)

        total = len(week_reminders)
        completed = sum(
            1 for r in week_reminders if r.get("status") in ("completed", "acknowledged")
        )
        missed = sum(1 for r in week_reminders if r.get("status") == "missed")
        completion_rate = completed / total if total > 0 else 0.0

        risk_scores = [
            r["cognitive_risk_score"]
            for r in week_reminders
            if r.get("cognitive_risk_score") is not None
        ]
        avg_risk = statistics.mean(risk_scores) if risk_scores else 0.0

        if avg_risk >= 0.75:
            risk_level = "critical"
        elif avg_risk >= 0.6:
            risk_level = "high"
        elif avg_risk >= 0.4:
            risk_level = "moderate"
        elif total > 0:
            risk_level = "low"
        else:
            risk_level = "unknown"

        alerts_col = Database.get_collection("caregiver_alerts")
        alert_count = await alerts_col.count_documents({
            "user_id": user_id,
            "created_at": {"$gte": start_dt, "$lte": end_dt},
        })
        critical_count = await alerts_col.count_documents({
            "user_id": user_id,
            "severity": "critical",
            "created_at": {"$gte": start_dt, "$lte": end_dt},
        })

        intervention_needed = avg_risk > 0.6 or (completion_rate < 0.5 and total > 0)

        return {
            "status": "success",
            "summary": {
                "user_id": user_id,
                "period": f"{start_dt.isoformat()} to {end_dt.isoformat()}",
                "risk_level": risk_level,
                "avg_cognitive_risk": round(avg_risk, 3),
                "risk_trend": "stable",
                "completion_rate": round(completion_rate, 3),
                "total_reminders": total,
                "completed": completed,
                "missed": missed,
                "total_alerts": alert_count,
                "critical_alerts": critical_count,
                "intervention_needed": intervention_needed,
                "escalation_required": avg_risk >= 0.75 or critical_count >= 3,
            }
        }

    except Exception as e:
        logger.error(f"Error generating report summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/complete/{reminder_id}", response_model=dict, status_code=status.HTTP_200_OK)
async def complete_reminder(reminder_id: str):
    """
    Mark a reminder as completed and handle repeat patterns.
    
    For daily/weekly reminders, this will:
    - Mark current instance as completed
    - Create next occurrence automatically
    
    - **reminder_id**: Reminder identifier
    """
    try:
        db_service = get_db_service()
        
        # Get the reminder
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        # Mark as completed
        completion_time = datetime.now()
        current_scheduled = reminder_data.get("scheduled_time")
        if isinstance(current_scheduled, str):
            current_scheduled = datetime.fromisoformat(current_scheduled.replace('+00:00', ''))

        # ── Log interaction so BehaviorTracker can learn from this completion ──
        # Prefer alarm_triggered_at (more accurate) over scheduled_time as base
        alarm_triggered_at = reminder_data.get("alarm_triggered_at")
        if alarm_triggered_at:
            if isinstance(alarm_triggered_at, str):
                alarm_triggered_at = datetime.fromisoformat(alarm_triggered_at.replace('+00:00', ''))
            response_time_seconds = (completion_time - alarm_triggered_at).total_seconds()
        elif current_scheduled:
            response_time_seconds = (completion_time - current_scheduled).total_seconds()
        else:
            response_time_seconds = None

        try:
            complete_interaction = ReminderInteraction(
                reminder_id=reminder_id,
                user_id=reminder_data["user_id"],
                reminder_category=reminder_data.get("category"),
                interaction_type=InteractionType.CONFIRMED,
                interaction_time=completion_time,
                response_time_seconds=response_time_seconds,
                user_response_text="Completed via /complete endpoint"
            )
            behavior_tracker.log_interaction(complete_interaction)
            logger.info(
                f"Logged CONFIRMED interaction for {reminder_id}: "
                f"response_time={response_time_seconds:.1f}s" if response_time_seconds else ""
            )
        except Exception as bt_err:
            logger.warning(f"BehaviorTracker log failed in complete: {bt_err}")

        await db_service.update_reminder(
            reminder_id,
            {
                "status": "completed",
                "completed_at": completion_time
            }
        )

        logger.info(f"Marked reminder {reminder_id} as completed")

        # Handle repeat patterns
        repeat_pattern = reminder_data.get("repeat_pattern")
        next_reminder_id = None
        next_scheduled_time = None
        adaptive_time_applied = 0

        if repeat_pattern in ["daily", "weekly"]:
            if repeat_pattern == "daily":
                next_scheduled_time = current_scheduled + timedelta(days=1)
            else:
                next_scheduled_time = current_scheduled + timedelta(weeks=1)

            # ── Adaptive time adjustment ─────────────────────────────────────
            # After ≥7 interactions, shift the alarm earlier/later based on the
            # user's measured average response delay.  Use a delta from the last
            # applied adjustment so the time never drifts unboundedly.
            if reminder_data.get("adaptive_scheduling_enabled", True):
                try:
                    pattern = behavior_tracker.get_user_behavior_pattern(
                        user_id=reminder_data["user_id"],
                        category=reminder_data.get("category"),
                        days=30
                    )
                    recommended = pattern.recommended_time_adjustment_minutes
                    # Delta from previously applied offset (prevents cumulative drift)
                    previously_applied = reminder_data.get("applied_time_adjustment_minutes", 0)
                    delta = recommended - previously_applied
                    if delta != 0 and abs(recommended) >= 5:  # only shift ≥5 min
                        next_scheduled_time += timedelta(minutes=delta)
                        adaptive_time_applied = recommended
                        logger.info(
                            f"Adaptive time shift applied for {reminder_data['user_id']}: "
                            f"recommended={recommended}min, delta={delta}min → "
                            f"next alarm at {next_scheduled_time.strftime('%H:%M')}"
                        )
                except Exception as adapt_err:
                    logger.warning(f"Adaptive scheduling lookup failed: {adapt_err}")

            # Create next reminder instance
            from src.features.reminder_system.reminder_models import Reminder, ReminderPriority, ReminderStatus

            next_reminder = Reminder(
                id=f"reminder_{uuid.uuid4().hex[:12]}",
                user_id=reminder_data["user_id"],
                title=reminder_data["title"],
                description=reminder_data.get("description"),
                scheduled_time=next_scheduled_time,
                priority=ReminderPriority(reminder_data.get("priority", "medium")),
                category=reminder_data.get("category", "general"),
                repeat_pattern=repeat_pattern,
                repeat_interval_minutes=reminder_data.get("repeat_interval_minutes"),
                caregiver_ids=reminder_data.get("caregiver_ids", []),
                adaptive_scheduling_enabled=reminder_data.get("adaptive_scheduling_enabled", True),
                escalation_enabled=reminder_data.get("escalation_enabled", True),
                escalation_threshold_minutes=reminder_data.get("escalation_threshold_minutes", 30),
                notify_caregiver_on_miss=reminder_data.get("notify_caregiver_on_miss", True),
                status=ReminderStatus.ACTIVE
            )

            result = await db_service.create_reminder(next_reminder)
            next_reminder_id = result["id"]

            # Persist the applied offset so next /complete can compute the delta
            if adaptive_time_applied != 0:
                await db_service.update_reminder(
                    next_reminder_id,
                    {"applied_time_adjustment_minutes": adaptive_time_applied}
                )

            logger.info(
                f"Created next {repeat_pattern} reminder {next_reminder_id} "
                f"scheduled for {next_scheduled_time}"
            )

        return {
            "status": "success",
            "message": "Reminder completed successfully",
            "reminder_id": reminder_id,
            "completed_at": completion_time.isoformat(),
            "response_time_seconds": round(response_time_seconds, 1) if response_time_seconds is not None else None,
            "adaptive_time_applied_minutes": adaptive_time_applied,
            "has_next_occurrence": next_reminder_id is not None,
            "next_reminder_id": next_reminder_id,
            "next_scheduled_time": next_scheduled_time.isoformat() if next_scheduled_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/acknowledge/{reminder_id}", response_model=dict, status_code=status.HTTP_200_OK)
async def acknowledge_reminder_alarm(
    reminder_id: str,
    user_id: str,
    acknowledgment_method: str = "tap"
):
    """
    Acknowledge a reminder alarm - user has explicitly confirmed they saw/heard it.
    
    This is REQUIRED to stop the alarm from repeating. The workflow is:
    1. Alarm is triggered (audio plays)
    2. User presses "Stop" button (alarm keeps playing)
    3. User must tap screen / say "OK" / press confirm (calls this endpoint)
    4. Alarm actually stops playing
    
    If this endpoint is not called within timeout:
    
    **Non-Escalation Mode (escalation_enabled=false):**
    - Alarm waits 3 minutes for acknowledgment (attempt 1)
    - If no response: alarm repeats after 3 minutes (attempt 2)
    - After 2 attempts (6 min total) → reminder marked as "missed" (no caregiver notification)
    
    **Escalation Mode (escalation_enabled=true):**
    - Alarm repeats every 3 minutes until escalation_threshold_minutes (default: 10 min)
    - After threshold exceeded → reminder marked as "missed" + caregiver notified
    - Multiple attempts for critical/high-priority reminders
    
    - **reminder_id**: Reminder identifier  
    - **user_id**: User who is acknowledging
    - **acknowledgment_method**: How they acknowledged (tap, verbal, button, gesture)
    """
    try:
        from src.features.reminder_system.realtime_engine import RealTimeReminderEngine
        
        # Note: In production, you'd get the engine instance from app state
        # For now, we'll update the database directly and log the interaction
        
        db_service = get_db_service()
        
        # Get the reminder
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        # Calculate response time if alarm was triggered
        now = datetime.now()
        response_time_seconds = None
        alarm_triggered_at = reminder_data.get("alarm_triggered_at")
        
        if alarm_triggered_at:
            if isinstance(alarm_triggered_at, str):
                alarm_triggered_at = datetime.fromisoformat(alarm_triggered_at.replace('+00:00', ''))
            response_time_seconds = (now - alarm_triggered_at).total_seconds()
        
        repeat_count = reminder_data.get("alarm_repeat_count", 0)
        
        response_time_msg = f" after {response_time_seconds:.1f}s" if response_time_seconds is not None else ""
        logger.info(
            f"✅ Reminder {reminder_id} acknowledged by user {user_id} "
            f"via {acknowledgment_method}{response_time_msg} "
            f"(repeated {repeat_count} times)"
        )
        
        # Update reminder status to acknowledged
        await db_service.update_reminder(
            reminder_id,
            {
                "status": "acknowledged",
                "acknowledged_at": now.isoformat(),
                "acknowledgment_method": acknowledgment_method,
                "response_time_seconds": response_time_seconds
            }
        )
        
        # Log the interaction for behavior tracking
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_category=reminder_data.get("category"),
            interaction_type=InteractionType.CONFIRMED,
            interaction_time=now,
            response_time_seconds=response_time_seconds,
            user_response_text=f"Acknowledged via {acknowledgment_method}"
        )
        behavior_tracker.log_interaction(interaction)
        
        # ── Behavioral Analysis: auto-log acknowledgment ──────────────────
        try:
            _bsvc = _get_behavioral_service()
            await _bsvc.log_behavioral_event(
                user_id=user_id,
                activity_type=ActivityType.REMINDER_RESPONSE,
                response_type=ResponseType.ACKNOWLEDGED,
                metadata={
                    "reminder_id": reminder_id,
                    "acknowledgment_method": acknowledgment_method,
                    "response_time_seconds": response_time_seconds,
                    "repeat_count": repeat_count
                }
            )
        except Exception as be:
            logger.warning(f"Behavioral logging failed: {be}")
        
        return {
            "status": "success",
            "message": "Alarm acknowledged - stopped successfully",
            "reminder_id": reminder_id,
            "acknowledgment_method": acknowledgment_method,
            "response_time_seconds": response_time_seconds,
            "repeat_count": repeat_count,
            "timestamp": now.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stop/{reminder_id}", response_model=dict, status_code=status.HTTP_200_OK)
async def stop_reminder_with_response(
    reminder_id: str,
    user_response: Optional[str] = None
):
    """
    Stop/Complete a reminder with optional user response for dementia tracking.
    
    This is the IMPROVED endpoint that:
    - Stops the alarm/notification
    - Analyzes user's response for dementia indicators
    - Tracks interaction to BehaviorTracker
    - Calculates cognitive risk score
    - Alerts caregiver if needed
    
    - **reminder_id**: Reminder identifier
    - **user_response**: User's response text (e.g., "done", "I took it", etc.)
    """
    try:
        db_service = get_db_service()
        
        # Get the reminder
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        completion_time = datetime.now()
        scheduled_time = reminder_data.get("scheduled_time")
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace('+00:00', ''))
        
        # Calculate response time
        response_time_seconds = None
        if scheduled_time:
            response_time_seconds = (completion_time - scheduled_time).total_seconds()
        
        # Analyze response if provided
        analysis_result = None
        cognitive_risk_score = 0.0
        interaction_type = InteractionType.CONFIRMED
        caregiver_notified = False
        
        if user_response and user_response.strip():
            # Analyze the response for dementia indicators
            from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
            
            reminder_obj = Reminder(
                id=reminder_id,
                user_id=reminder_data["user_id"],
                title=reminder_data["title"],
                description=reminder_data.get("description"),
                scheduled_time=scheduled_time or datetime.now(),
                priority=ReminderPriority(reminder_data.get("priority", "medium")),
                category=reminder_data.get("category", "general")
            )
            
            result = scheduler.process_reminder_response(
                reminder=reminder_obj,
                user_response=user_response,
                response_time_seconds=response_time_seconds
            )
            
            analysis_result = result.get('analysis', {})
            cognitive_risk_score = analysis_result.get('cognitive_risk_score', 0.0)
            interaction_type = InteractionType(analysis_result.get('interaction_type', 'confirmed'))
            caregiver_notified = result.get('caregiver_notified', False)
        else:
            # No response provided - just mark as completed
            # Still track interaction for behavior analysis
            interaction = ReminderInteraction(
                reminder_id=reminder_id,
                user_id=reminder_data["user_id"],
                reminder_category=reminder_data.get("category"),
                interaction_type=InteractionType.CONFIRMED,
                interaction_time=completion_time,
                response_time_seconds=response_time_seconds
            )
            behavior_tracker.log_interaction(interaction)
        
        # Mark as completed
        await db_service.update_reminder(
            reminder_id,
            {
                "status": "completed",
                "completed_at": completion_time,
                "user_response": user_response,
                "cognitive_risk_score": cognitive_risk_score
            }
        )
        
        logger.info(f"Stopped reminder {reminder_id} - cognitive risk: {cognitive_risk_score:.2f}")

        # ── Behavioral Analysis: auto-log this interaction ──────────────────
        try:
            _bsvc = _get_behavioral_service()
            _deviation_mins = None
            if scheduled_time and response_time_seconds is not None:
                _deviation_mins = response_time_seconds / 60.0

            # Map interaction_type to ResponseType
            _resp_map = {
                InteractionType.CONFIRMED:         ResponseType.RESPONDED_ON_TIME,
                InteractionType.DELAYED:           ResponseType.RESPONDED_LATE,
                InteractionType.IGNORED:           ResponseType.MISSED,
                InteractionType.CONFUSED:          ResponseType.CONFUSED,
                InteractionType.REPEATED_QUESTION: ResponseType.CONFUSED,
                InteractionType.PARTIAL_COMPLETION:ResponseType.RESPONDED_LATE,
            }
            _rt = _resp_map.get(interaction_type, ResponseType.RESPONDED_ON_TIME)
            if _deviation_mins and _deviation_mins > 30:
                _rt = ResponseType.RESPONDED_LATE

            _log = UserBehavioralLog(
                user_id=reminder_data["user_id"],
                activity_type=ActivityType.MEDICATION
                    if reminder_data.get("category") == "medication"
                    else ActivityType.REMINDER_RESPONSE,
                response_type=_rt,
                timestamp=completion_time,
                scheduled_time=scheduled_time,
                time_deviation_minutes=_deviation_mins,
                completion_rate=1.0 if interaction_type == InteractionType.CONFIRMED else 0.5,
                reminder_id=reminder_id,
                reminder_category=reminder_data.get("category"),
            )
            import asyncio
            asyncio.create_task(_bsvc.log_event(_log))
        except Exception as _be:
            logger.warning(f"Behavioral log skipped: {_be}")
        # ────────────────────────────────────────────────────────────────────

        # Handle repeat patterns
        repeat_pattern = reminder_data.get("repeat_pattern")
        next_reminder_id = None
        next_scheduled_time = None
        
        if repeat_pattern in ["daily", "weekly"]:
            current_scheduled = reminder_data["scheduled_time"]
            if isinstance(current_scheduled, str):
                current_scheduled = datetime.fromisoformat(current_scheduled.replace('+00:00', ''))
            
            if repeat_pattern == "daily":
                next_scheduled_time = current_scheduled + timedelta(days=1)
            elif repeat_pattern == "weekly":
                next_scheduled_time = current_scheduled + timedelta(weeks=1)
            
            # Create next reminder instance
            from src.features.reminder_system.reminder_models import Reminder, ReminderPriority, ReminderStatus
            
            next_reminder = Reminder(
                id=f"reminder_{uuid.uuid4().hex[:12]}",
                user_id=reminder_data["user_id"],
                title=reminder_data["title"],
                description=reminder_data.get("description"),
                scheduled_time=next_scheduled_time,
                priority=ReminderPriority(reminder_data.get("priority", "medium")),
                category=reminder_data.get("category", "general"),
                repeat_pattern=repeat_pattern,
                repeat_interval_minutes=reminder_data.get("repeat_interval_minutes"),
                caregiver_ids=reminder_data.get("caregiver_ids", []),
                adaptive_scheduling_enabled=reminder_data.get("adaptive_scheduling_enabled", True),
                escalation_enabled=reminder_data.get("escalation_enabled", True),
                escalation_threshold_minutes=reminder_data.get("escalation_threshold_minutes", 30),
                notify_caregiver_on_miss=reminder_data.get("notify_caregiver_on_miss", True),
                status=ReminderStatus.ACTIVE
            )
            
            result = await db_service.create_reminder(next_reminder)
            next_reminder_id = result["id"]
            
            logger.info(f"Created next {repeat_pattern} reminder {next_reminder_id}")
        
        return {
            "status": "success",
            "message": "Reminder stopped successfully",
            "reminder_id": reminder_id,
            "completed_at": completion_time.isoformat(),
            "cognitive_analysis": {
                "risk_score": cognitive_risk_score,
                "interaction_type": interaction_type.value,
                "features": analysis_result.get('features', {}) if analysis_result else {},
                "caregiver_notified": caregiver_notified,
                "confusion_detected": analysis_result.get('confusion_detected', False) if analysis_result else False,
                "memory_issue_detected": analysis_result.get('memory_issue_detected', False) if analysis_result else False
            },
            "has_next_occurrence": next_reminder_id is not None,
            "next_reminder_id": next_reminder_id,
            "next_scheduled_time": next_scheduled_time.isoformat() if next_scheduled_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/missed/{reminder_id}", response_model=dict, status_code=status.HTTP_200_OK)
async def mark_reminder_missed(reminder_id: str, user_id: Optional[str] = None):
    """
    Mark a reminder as missed from the frontend local timeout cycle.

    Called by the mobile/web frontend when the user ignores all repeats and the
    frontend-owned alarm timeout cycle has exhausted MAX_REPEATS with no
    acknowledgment.  This keeps the database status in sync when the backend's
    own realtime engine has not yet marked the reminder as missed (e.g. because
    the WebSocket connection was lost or the frontend cycle fired first).

    - **reminder_id**: Reminder identifier
    - **user_id**: Optional user ID for behavior logging
    """
    try:
        db_service = get_db_service()

        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )

        # Idempotent — if it's already missed/acknowledged, return success without
        # overwriting the existing status.
        current_status = reminder_data.get("status", "")
        if current_status in ("missed", "acknowledged", "completed"):
            return {
                "status": "success",
                "message": f"Reminder already in terminal state: {current_status}",
                "reminder_id": reminder_id,
                "current_status": current_status
            }

        missed_at = datetime.now()
        effective_user_id = user_id or reminder_data.get("user_id", "unknown")

        await db_service.update_reminder(
            reminder_id,
            {
                "status": "missed",
                "missed_at": missed_at.isoformat(),
            }
        )

        # Log the ignored interaction for behavior / dementia tracking
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=effective_user_id,
            reminder_category=reminder_data.get("category"),
            interaction_type=InteractionType.IGNORED,
            interaction_time=missed_at,
        )
        behavior_tracker.log_interaction(interaction)

        try:
            _bsvc = _get_behavioral_service()
            _log = UserBehavioralLog(
                user_id=effective_user_id,
                activity_type=ActivityType.MEDICATION
                    if reminder_data.get("category") == "medication"
                    else ActivityType.REMINDER_RESPONSE,
                response_type=ResponseType.MISSED,
                timestamp=missed_at,
                completion_rate=0.0,
                reminder_id=reminder_id,
                reminder_category=reminder_data.get("category"),
            )
            import asyncio
            asyncio.create_task(_bsvc.log_event(_log))
        except Exception as _be:
            logger.warning(f"Behavioral log skipped for missed reminder: {_be}")

        logger.info(
            f"⚠️ Reminder {reminder_id} marked as missed by frontend timeout cycle "
            f"(user: {effective_user_id})"
        )

        # ── Alert caregivers when a medication reminder is missed ──────────
        caregiver_ids = reminder_data.get("caregiver_ids", [])
        category      = reminder_data.get("category", "general")
        priority_val  = reminder_data.get("priority", "medium")
        if caregiver_ids and category == "medication":
            sched_raw = reminder_data.get("scheduled_time")
            sched_dt  = (
                datetime.fromisoformat(sched_raw.replace("+00:00", ""))
                if isinstance(sched_raw, str) else sched_raw
            )
            await _save_caregiver_alert(
                patient_id=effective_user_id,
                caregiver_ids=caregiver_ids,
                alert_type="missed_medication",
                priority=priority_val,
                reminder_id=reminder_id,
                reminder_title=reminder_data.get("title", "Reminder"),
                reminder_category=category,
                scheduled_time=sched_dt,
            )

        return {
            "status": "success",
            "message": "Reminder marked as missed",
            "reminder_id": reminder_id,
            "missed_at": missed_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking reminder as missed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/snooze-tracked/{reminder_id}", response_model=dict)
async def snooze_reminder_with_tracking(reminder_id: str, delay_minutes: int = 15):
    """
    Snooze a reminder and track the interaction for dementia detection.
    
    This IMPROVED snooze endpoint:
    - Delays the reminder by specified minutes
    - Tracks snooze behavior to BehaviorTracker
    - Monitors frequent snoozing patterns (may indicate confusion/avoidance)
    - Updates adaptive scheduling
    
    - **reminder_id**: Reminder identifier
    - **delay_minutes**: Minutes to delay (default 15)
    """
    try:
        db_service = get_db_service()
        
        # Get reminder from database
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        scheduled_time = reminder_data.get("scheduled_time")
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace('+00:00', ''))
        
        reminder = Reminder(
            id=reminder_id,
            user_id=reminder_data["user_id"],
            title=reminder_data["title"],
            scheduled_time=scheduled_time or datetime.now(),
            priority=ReminderPriority(reminder_data.get("priority", "medium")),
            category=reminder_data.get("category", "general")
        )
        
        # Reschedule the reminder
        updated_reminder = scheduler.reschedule_reminder(
            reminder=reminder,
            delay_minutes=delay_minutes,
            reason="user_requested_snooze"
        )
        
        # Track the snooze interaction
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=reminder_data["user_id"],
            reminder_category=reminder_data.get("category"),
            interaction_type=InteractionType.DELAYED,
            interaction_time=datetime.now(),
            user_response_text=f"Snoozed for {delay_minutes} minutes"
        )
        behavior_tracker.log_interaction(interaction)
        
        # Check snooze frequency (frequent snoozing may indicate confusion)
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=reminder_data["user_id"],
            reminder_id=reminder_id,
            days=7
        )
        
        # Alert if too many snoozes
        alert_caregiver = False
        if pattern.delayed_count > 5 and pattern.total_reminders > 0:
            delay_rate = pattern.delayed_count / pattern.total_reminders
            if delay_rate > 0.6:  # More than 60% snooze rate
                alert_caregiver = True
                logger.warning(f"High snooze rate for user {reminder_data['user_id']}: {delay_rate:.1%}")
        
        # Update reminder in database
        await db_service.update_reminder(
            reminder_id,
            {
                "scheduled_time": updated_reminder.scheduled_time,
                "status": "snoozed"
            }
        )
        
        return {
            "status": "success",
            "message": f"Reminder snoozed for {delay_minutes} minutes",
            "reminder_id": reminder_id,
            "new_scheduled_time": updated_reminder.scheduled_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "snooze_count_week": pattern.delayed_count,
            "snooze_rate": pattern.delayed_count / pattern.total_reminders if pattern.total_reminders > 0 else 0,
            "caregiver_alert": alert_caregiver,
            "recommendation": "Consider discussing with caregiver" if alert_caregiver else "Normal snooze behavior"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error snoozing reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/help-request/{reminder_id}", response_model=dict)
async def request_help_confused(
    reminder_id: str,
    help_reason: Optional[str] = "confused"
):
    """
    User requests help or indicates confusion about a reminder.
    
    This endpoint is triggered when user indicates they:
    - Don't understand the reminder
    - Are confused about what to do
    - Need assistance from caregiver
    
    This is a CRITICAL indicator for dementia detection and triggers immediate caregiver alert.
    
    - **reminder_id**: Reminder identifier
    - **help_reason**: Reason for help (confused, dont_understand, forgot, need_assistance)
    """
    try:
        db_service = get_db_service()
        
        # Get reminder
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        user_id = reminder_data["user_id"]
        
        # Track as confused interaction - HIGH cognitive risk indicator
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_category=reminder_data.get("category"),
            interaction_type=InteractionType.CONFUSED,
            interaction_time=datetime.now(),
            user_response_text=f"Help requested: {help_reason}",
            cognitive_risk_score=0.75,  # High risk - explicit confusion
            confusion_detected=True,
            recommended_action="escalate_to_caregiver",
            caregiver_alert_triggered=True
        )
        behavior_tracker.log_interaction(interaction)
        
        # Update reminder status
        await db_service.update_reminder(
            reminder_id,
            {
                "status": "requires_help",
                "help_requested": True,
                "help_reason": help_reason,
                "help_requested_at": datetime.now()
            }
        )
        
        # Trigger caregiver notification
        from src.features.reminder_system.caregiver_notifier import CaregiverNotifier
        from src.features.reminder_system.reminder_models import CaregiverAlert
        
        notifier = CaregiverNotifier()
        
        # Create alert for all assigned caregivers
        caregiver_ids = reminder_data.get("caregiver_ids", [])
        alerts_sent = []
        
        for caregiver_id in caregiver_ids:
            alert = CaregiverAlert(
                id=f"alert_{uuid.uuid4().hex[:12]}",
                caregiver_id=caregiver_id,
                user_id=user_id,
                reminder_id=reminder_id,
                alert_type="confusion_detected",
                severity="high",
                message=f"Patient is confused about: {reminder_data['title']}",
                reminder_title=reminder_data["title"],
                cognitive_risk_score=0.75,
                user_response=f"Help requested: {help_reason}",
                context={
                    "help_reason": help_reason,
                    "category": reminder_data.get("category"),
                    "priority": reminder_data.get("priority")
                }
            )
            
            notifier.send_alert(alert)
            alerts_sent.append(caregiver_id)
        
        logger.warning(f"HELP REQUESTED: User {user_id} confused about reminder {reminder_id}")
        logger.info(f"Caregiver alerts sent to: {alerts_sent}")
        
        return {
            "status": "success",
            "message": "Help request received. Caregiver has been notified.",
            "reminder_id": reminder_id,
            "cognitive_risk_score": 0.75,
            "caregiver_notified": True,
            "caregivers_alerted": alerts_sent,
            "recommendation": "Caregiver will contact you shortly",
            "help_reason": help_reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing help request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/weekly-dementia-risk/{user_id}", response_model=dict)
async def get_weekly_dementia_risk_report(
    user_id: str,
    weeks: int = 4
):
    """
    Generate weekly dementia risk report for caregiver dashboard.
    
    Provides comprehensive cognitive decline tracking over multiple weeks:
    - Average cognitive risk scores per week
    - Trend analysis (improving, stable, declining)
    - Confusion frequency tracking
    - Response pattern analysis
    - Recommendations for intervention
    
    - **user_id**: Patient identifier
    - **weeks**: Number of weeks to analyze (default 4)
    """
    try:
        # Get behavior patterns for each week
        weekly_data = []
        
        for week in range(weeks):
            start_day = week * 7
            end_day = start_day + 7
            
            pattern = behavior_tracker.get_user_behavior_pattern(
                user_id=user_id,
                days=7  # One week at a time
            )
            
            week_number = weeks - week  # Most recent week is 1
            
            weekly_data.append({
                "week": week_number,
                "total_reminders": pattern.total_reminders,
                "confirmed_count": pattern.confirmed_count,
                "confused_count": pattern.confused_count,
                "ignored_count": pattern.ignored_count,
                "avg_cognitive_risk": pattern.avg_cognitive_risk_score or 0.0,
                "confusion_trend": pattern.confusion_trend,
                "escalation_needed": pattern.escalation_recommended
            })
        
        # Calculate overall trends
        risk_scores = [w["avg_cognitive_risk"] for w in weekly_data if w["avg_cognitive_risk"] > 0]
        avg_risk = statistics.mean(risk_scores) if risk_scores else 0.0
        
        # Determine trend
        if len(risk_scores) >= 2:
            recent_avg = statistics.mean(risk_scores[:2])  # Last 2 weeks
            older_avg = statistics.mean(risk_scores[2:]) if len(risk_scores) > 2 else recent_avg
            
            if recent_avg > older_avg + 0.1:
                overall_trend = "declining"  # Cognitive function declining
                trend_severity = "concerning"
            elif recent_avg < older_avg - 0.1:
                overall_trend = "improving"
                trend_severity = "positive"
            else:
                overall_trend = "stable"
                trend_severity = "normal"
        else:
            overall_trend = "insufficient_data"
            trend_severity = "unknown"
        
        # Generate recommendations
        recommendations = []
        if avg_risk > 0.7:
            recommendations.append("⚠️ HIGH RISK: Schedule medical evaluation immediately")
        elif avg_risk > 0.5:
            recommendations.append("📋 Elevated risk: Increase monitoring frequency")
        
        total_confused = sum(w["confused_count"] for w in weekly_data)
        if total_confused > 10:
            recommendations.append("🧠 High confusion frequency - consider cognitive assessment")
        
        if overall_trend == "declining":
            recommendations.append("📉 Cognitive decline detected - consult healthcare provider")
        
        if not recommendations:
            recommendations.append("✅ Cognitive function appears stable")
        
        logger.info(f"Generated {weeks}-week dementia risk report for user {user_id}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "report_period_weeks": weeks,
            "summary": {
                "average_cognitive_risk": avg_risk,
                "overall_trend": overall_trend,
                "trend_severity": trend_severity,
                "total_confused_interactions": total_confused,
                "requires_immediate_attention": avg_risk > 0.7 or overall_trend == "declining"
            },
            "weekly_breakdown": weekly_data,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint for reminder system."""
    return {
        "status": "healthy",
        "service": "reminder_system",
        "timestamp": datetime.now().isoformat()
    }


# ========================================================================
# TESTING & DEBUGGING ENDPOINTS - Test Adaptive Scheduling
# ========================================================================

@router.post("/test/simulate-interactions", response_model=dict)
async def simulate_user_interactions(
    user_id: str,
    num_interactions: int = 10,
    category: str = "medication",
    base_delay_minutes: int = 0
):
    """
    🧪 TEST ENDPOINT: Simulate user interactions to test adaptive scheduling.
    
    Creates fake reminder interactions with realistic response patterns.
    Use this to build up behavior data for testing adaptive scheduling.
    
    **Parameters:**
    - **user_id**: User identifier
    - **num_interactions**: Number of interactions to simulate (default: 10)
    - **category**: Reminder category (medication, meal, hygiene, etc.)
    - **base_delay_minutes**: Base response delay in minutes (0 = on time, positive = late)
    
    **Example scenarios:**
    - `base_delay_minutes=0`: User always on time
    - `base_delay_minutes=30`: User consistently 30 minutes late
    - `base_delay_minutes=-15`: User responds 15 minutes early
    """
    try:
        import random
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        interactions_created = []
        
        for i in range(num_interactions):
            # Create reminder at different times
            hour = random.choice([7, 8, 9, 12, 13, 18, 20, 21])
            days_ago = random.randint(1, 30)
            
            scheduled_time = datetime.now() - timedelta(days=days_ago)
            scheduled_time = scheduled_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Simulate response with variation
            response_delay = base_delay_minutes + random.randint(-10, 10)
            response_time = scheduled_time + timedelta(minutes=max(0, response_delay))
            
            # Determine interaction type
            if response_delay < -5:
                interaction_type = InteractionType.CONFIRMED  # Responded early
            elif response_delay < 30:
                interaction_type = InteractionType.CONFIRMED  # On time
            elif response_delay < 60:
                interaction_type = InteractionType.DELAYED  # Snoozed
            else:
                interaction_type = InteractionType.IGNORED  # Too late/ignored
            
            # Occasionally add confusion
            if random.random() < 0.1:  # 10% confusion rate
                interaction_type = InteractionType.CONFUSED
            
            # Create interaction
            interaction = ReminderInteraction(
                reminder_id=f"test_reminder_{i}",
                user_id=user_id,
                reminder_category=category,
                interaction_type=interaction_type,
                interaction_time=response_time,
                response_time_seconds=response_delay * 60,
                user_response_text=f"Simulated response {i}",
                cognitive_risk_score=random.uniform(0.1, 0.4) if interaction_type != InteractionType.CONFUSED else random.uniform(0.6, 0.9)
            )
            
            # Log to behavior tracker
            behavior_tracker.log_interaction(interaction)
            
            interactions_created.append({
                "scheduled_time": scheduled_time.isoformat(),
                "response_time": response_time.isoformat(),
                "delay_minutes": response_delay,
                "interaction_type": interaction_type.value,
                "hour": hour
            })
        
        # Get updated behavior pattern
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            category=category,
            days=30
        )
        
        logger.info(f"✅ Simulated {num_interactions} interactions for user {user_id}")
        
        return {
            "status": "success",
            "message": f"Simulated {num_interactions} interactions",
            "user_id": user_id,
            "interactions_created": interactions_created,
            "learned_pattern": {
                "optimal_hour": pattern.optimal_reminder_hour,
                "worst_hours": pattern.worst_response_hours,
                "recommended_time_adjustment": pattern.recommended_time_adjustment_minutes,
                "frequency_multiplier": pattern.recommended_frequency_multiplier,
                "confusion_trend": pattern.confusion_trend,
                "total_interactions": pattern.total_reminders
            }
        }
        
    except Exception as e:
        logger.error(f"Error simulating interactions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/test/adaptive-schedule", response_model=dict)
async def test_adaptive_scheduling(
    user_id: str,
    scheduled_hour: int = 8,
    category: str = "medication",
    priority: str = "medium"
):
    """
    🧪 TEST ENDPOINT: Test adaptive scheduling recommendations.
    
    Creates a test reminder and gets adaptive scheduling recommendations
    based on learned user behavior patterns.
    
    **Parameters:**
    - **user_id**: User identifier (must have interaction history)
    - **scheduled_hour**: Hour to schedule reminder (0-23)
    - **category**: Reminder category
    - **priority**: Reminder priority (low, medium, high, critical)
    
    **Returns:**
    - Optimal scheduling time based on learned patterns
    - Frequency recommendations
    - Urgency level
    - Detailed behavior statistics
    """
    try:
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        # Create test reminder
        scheduled_time = datetime.now().replace(hour=scheduled_hour, minute=0, second=0, microsecond=0)
        if scheduled_time < datetime.now():
            scheduled_time += timedelta(days=1)
        
        reminder = Reminder(
            id=f"test_reminder_{datetime.now().timestamp()}",
            user_id=user_id,
            title=f"Test {category.capitalize()} Reminder",
            description="Testing adaptive scheduling",
            scheduled_time=scheduled_time,
            priority=ReminderPriority(priority),
            category=category,
            status=ReminderStatus.ACTIVE
        )
        
        # Get adaptive schedule recommendation
        schedule_info = scheduler.get_optimal_reminder_schedule(reminder, days_analysis=30)
        
        # Get behavior pattern
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            category=category,
            days=30
        )
        
        # Test if should send now
        should_send, reason = scheduler.should_send_reminder_now(reminder)
        
        return {
            "status": "success",
            "message": "Adaptive scheduling analysis complete",
            "input": {
                "user_id": user_id,
                "requested_time": scheduled_time.isoformat(),
                "requested_hour": scheduled_hour,
                "category": category,
                "priority": priority
            },
            "adaptive_recommendations": {
                "optimal_time": schedule_info['optimal_time'].isoformat() if schedule_info.get('optimal_time') else None,
                "time_adjustment_minutes": schedule_info.get('time_adjustment_minutes', 0),
                "frequency_multiplier": schedule_info.get('frequency_multiplier', 1.0),
                "urgency_level": schedule_info.get('urgency_level', 'normal'),
                "escalation_needed": schedule_info.get('escalation_needed', False),
                "worst_hours_to_avoid": schedule_info.get('worst_hours', [])
            },
            "behavior_stats": {
                "total_reminders": pattern.total_reminders,
                "confirmed_count": pattern.confirmed_count,
                "ignored_count": pattern.ignored_count,
                "delayed_count": pattern.delayed_count,
                "confused_count": pattern.confused_count,
                "confirmation_rate": pattern.confirmed_count / pattern.total_reminders if pattern.total_reminders > 0 else 0,
                "confusion_rate": pattern.confused_count / pattern.total_reminders if pattern.total_reminders > 0 else 0,
                "optimal_reminder_hour": pattern.optimal_reminder_hour,
                "avg_cognitive_risk_score": pattern.avg_cognitive_risk_score,
                "confusion_trend": pattern.confusion_trend
            },
            "should_send_now": {
                "send": should_send,
                "reason": reason
            },
            "pattern_details": schedule_info.get('pattern_stats', {})
        }
        
    except Exception as e:
        logger.error(f"Error testing adaptive scheduling: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/test/behavior-analysis/{user_id}", response_model=dict)
async def get_detailed_behavior_analysis(
    user_id: str,
    days: int = 30,
    category: Optional[str] = None
):
    """
    🧪 TEST ENDPOINT: Get detailed behavior analysis for a user.
    
    Shows comprehensive analytics about user's reminder interaction patterns,
    helping verify that adaptive scheduling is learning correctly.
    
    **Parameters:**
    - **user_id**: User identifier
    - **days**: Analysis period in days (default: 30)
    - **category**: Optional category filter
    
    **Returns:**
    - Complete behavior pattern analysis
    - Time-of-day breakdown
    - Trend analysis
    - Recommendations
    """
    try:
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            category=category,
            days=days
        )
        
        # Calculate additional metrics
        total = pattern.total_reminders
        
        return {
            "status": "success",
            "user_id": user_id,
            "analysis_period_days": days,
            "category_filter": category,
            "summary": {
                "total_interactions": total,
                "has_sufficient_data": total >= 10,
                "data_quality": "excellent" if total >= 30 else "good" if total >= 10 else "limited"
            },
            "interaction_breakdown": {
                "confirmed": pattern.confirmed_count,
                "ignored": pattern.ignored_count,
                "delayed": pattern.delayed_count,
                "confused": pattern.confused_count,
                "confirmation_rate": pattern.confirmed_count / total if total > 0 else 0,
                "ignore_rate": pattern.ignored_count / total if total > 0 else 0,
                "delay_rate": pattern.delayed_count / total if total > 0 else 0,
                "confusion_rate": pattern.confused_count / total if total > 0 else 0
            },
            "timing_analysis": {
                "optimal_reminder_hour": pattern.optimal_reminder_hour,
                "worst_response_hours": pattern.worst_response_hours,
                "avg_response_time_seconds": pattern.avg_response_time_seconds,
                "avg_response_time_minutes": pattern.avg_response_time_seconds / 60 if pattern.avg_response_time_seconds else None
            },
            "cognitive_health": {
                "avg_cognitive_risk_score": pattern.avg_cognitive_risk_score,
                "confusion_trend": pattern.confusion_trend,
                "escalation_recommended": pattern.escalation_recommended,
                "trend_interpretation": {
                    "improving": "User's confusion rate is decreasing over time ✅",
                    "stable": "User's confusion rate is consistent 👍",
                    "declining": "User's confusion rate is increasing - caregiver alert needed ⚠️",
                    "insufficient_data": "Not enough data to determine trend"
                }.get(pattern.confusion_trend, "Unknown")
            },
            "adaptive_recommendations": {
                "frequency_multiplier": pattern.recommended_frequency_multiplier,
                "frequency_interpretation": {
                    1.5: "Increase reminder frequency by 50% (poor completion rate)",
                    1.3: "Increase reminder frequency by 30% (some missed reminders)",
                    1.0: "Maintain current reminder frequency",
                    0.9: "Can reduce reminder frequency by 10% (excellent completion)"
                }.get(pattern.recommended_frequency_multiplier, "Maintain current frequency"),
                "time_adjustment_minutes": pattern.recommended_time_adjustment_minutes,
                "time_adjustment_interpretation": 
                    f"Shift reminders {abs(pattern.recommended_time_adjustment_minutes)} minutes {'earlier' if pattern.recommended_time_adjustment_minutes < 0 else 'later'}" 
                    if pattern.recommended_time_adjustment_minutes else "No time adjustment needed"
            },
            "raw_pattern_data": pattern.dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting behavior analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/test/scheduler-health", response_model=dict)
async def get_scheduler_health():
    """
    🧪 TEST ENDPOINT: Check adaptive scheduler health and statistics.
    
    Returns overall statistics about the adaptive scheduling system,
    including how many users have behavior data, system status, etc.
    
    **Returns:**
    - System health status
    - User statistics
    - Scheduler configuration
    """
    try:
        return {
            "status": "healthy",
            "message": "Adaptive scheduler is operational",
            "scheduler_info": {
                "type": "rule_based",
                "ml_model_available": False,  # Will be True when ML model is added
                "features": [
                    "Time-of-day optimization",
                    "Worst hour avoidance",
                    "Response time tracking",
                    "Confusion trend analysis",
                    "Frequency adjustment",
                    "Cognitive risk monitoring",
                    "Caregiver escalation"
                ]
            },
            "behavior_tracker": {
                "active": True,
                "cached_users": len(behavior_tracker.interaction_cache),
                "tracking_metrics": [
                    "Confirmation rate",
                    "Ignore rate",
                    "Delay rate",
                    "Confusion rate",
                    "Cognitive risk scores",
                    "Response times"
                ]
            },
            "testing_tips": {
                "step_1": "POST /api/reminders/test/simulate-interactions - Create test data",
                "step_2": "GET /api/reminders/test/behavior-analysis/{user_id} - View learned patterns",
                "step_3": "POST /api/reminders/test/adaptive-schedule - Test scheduling recommendations",
                "step_4": "Create real reminder and see adaptive time applied"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler health: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
