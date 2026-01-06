"""
BERT-based Text Parser for Reminder Extraction

Uses transformer models to extract:
- Dates and times
- Medical entities (medications, conditions)
- Actions and intents
- More robust than regex patterns
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)

# Try to import transformers, fall back to regex if not available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    logger.warning("dateparser not available. Install with: pip install dateparser")


class BERTReminderParser:
    """
    BERT-based parser for extracting reminder information from natural language.
    """
    
    def __init__(self):
        """Initialize BERT models for NER and date parsing."""
        self.ner_model = None
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load pre-trained NER model (English)
                logger.info("Loading BERT NER model...")
                self.ner_model = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                logger.info("✅ BERT NER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BERT model: {e}")
                self.ner_model = None
        
        # Medical terms dictionary for category detection
        self.medical_terms = {
            "medication": [
                "medicine", "medication", "pill", "tablet", "drug", "dose",
                "prescription", "meds", "aspirin", "ibuprofen", "antibiotic",
                "insulin", "inhaler", "vitamin", "supplement"
            ],
            "appointment": [
                "doctor", "appointment", "visit", "checkup", "clinic",
                "hospital", "physician", "dentist", "therapy", "consultation"
            ],
            "meal": [
                "breakfast", "lunch", "dinner", "meal", "eat", "food",
                "snack", "brunch", "supper"
            ],
            "hygiene": [
                "shower", "bath", "hygiene", "brush", "wash", "clean",
                "shave", "grooming"
            ],
            "activity": [
                "exercise", "walk", "activity", "gym", "yoga", "swim",
                "run", "workout", "physical therapy"
            ]
        }
    
    def parse_reminder(
        self, 
        text: str, 
        user_id: str,
        priority_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse reminder text using BERT + dateparser for better accuracy.
        
        Args:
            text: Natural language reminder text
            user_id: User identifier
            priority_override: Optional priority override
            
        Returns:
            Dictionary with parsed reminder details
        """
        text_lower = text.lower()
        
        # Extract entities using BERT if available
        entities = self._extract_entities_bert(text) if self.ner_model else []
        
        # Extract category
        category = self._extract_category(text_lower, entities)
        
        # Extract priority
        priority = self._extract_priority(text_lower, priority_override)
        
        # Extract date and time using dateparser (more powerful than regex)
        scheduled_time = self._extract_datetime(text, text_lower)
        
        # Extract recurrence pattern
        recurrence = self._extract_recurrence(text_lower)
        
        # Generate title
        title = self._generate_title(text, category, entities)
        
        # Extract medication names if category is medication
        medication_names = []
        if category == "medication":
            medication_names = self._extract_medication_names(text, entities)
        
        return {
            "title": title,
            "description": text,
            "category": category,
            "priority": priority,
            "scheduled_time": scheduled_time,
            "recurrence": recurrence,
            "entities": entities,
            "medication_names": medication_names
        }
    
    def _extract_entities_bert(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using BERT NER model."""
        if not self.ner_model:
            return []
        
        try:
            entities = self.ner_model(text)
            logger.info(f"BERT extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"BERT entity extraction error: {e}")
            return []
    
    def _extract_category(self, text_lower: str, entities: List[Dict]) -> str:
        """
        Extract category using keyword matching + BERT entities.
        """
        # Check for medical-related entities from BERT
        for entity in entities:
            entity_type = entity.get('entity_group', '').upper()
            if entity_type in ['MISC', 'ORG']:
                word = entity.get('word', '').lower()
                # Check if it's a medical term
                for cat, terms in self.medical_terms.items():
                    if any(term in word for term in terms):
                        return cat
        
        # Fallback to keyword matching
        for category, keywords in self.medical_terms.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return "general"
    
    def _extract_priority(self, text_lower: str, override: Optional[str]) -> str:
        """Extract priority level."""
        if override:
            return override
        
        high_priority_words = ["urgent", "critical", "important", "asap", "emergency", "immediately"]
        low_priority_words = ["when you can", "sometime", "eventually", "optional"]
        
        if any(word in text_lower for word in high_priority_words):
            return "high"
        elif any(word in text_lower for word in low_priority_words):
            return "low"
        
        return "medium"
    
    def _extract_datetime(self, text: str, text_lower: str) -> datetime:
        """
        Extract date and time using dateparser (handles natural language).
        Falls back to manual parsing if dateparser not available.
        """
        now = datetime.now()
        
        # Try dateparser first (best option - handles natural language)
        if DATEPARSER_AVAILABLE:
            try:
                # dateparser can understand: "tomorrow at 6pm", "next monday 2pm", etc.
                parsed_dt = dateparser.parse(
                    text,
                    settings={
                        'PREFER_DATES_FROM': 'future',
                        'RETURN_AS_TIMEZONE_AWARE': False,
                        'RELATIVE_BASE': now
                    }
                )
                
                if parsed_dt and parsed_dt > now:
                    logger.info(f"dateparser extracted: {parsed_dt}")
                    return parsed_dt
            except Exception as e:
                logger.error(f"dateparser error: {e}")
        
        # Fallback to manual parsing
        return self._manual_datetime_extraction(text_lower, now)
    
    def _manual_datetime_extraction(self, text_lower: str, now: datetime) -> datetime:
        """
        Manual date/time extraction (fallback).
        This is the enhanced version from the previous fix.
        """
        # Extract DATE
        days_offset = 0
        
        if "tomorrow" in text_lower:
            days_offset = 1
        elif "day after tomorrow" in text_lower:
            days_offset = 2
        elif "next week" in text_lower:
            days_offset = 7
        
        # Weekdays
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in weekdays.items():
            if day_name in text_lower:
                current_weekday = now.weekday()
                days_until = (day_num - current_weekday) % 7
                if days_until == 0:
                    days_until = 7
                days_offset = days_until
                break
        
        # Extract TIME
        hour = None
        minute = 0
        time_found = False
        
        time_patterns = [
            (r'(\d{1,2})\s*[:\.]\s*(\d{2})\s*(am|pm|a\.m\.|p\.m\.)', 3),
            (r'(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)', 2),
            (r'(\d{1,2})\s*[:\.]\s*(\d{2})\b', 2),
            (r'at\s+(\d{1,2})(?:\s*o\'?clock)?', 1),
        ]
        
        for pattern, group_count in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    groups = match.groups()
                    if group_count == 3:
                        hour = int(groups[0])
                        minute = int(groups[1])
                        period = groups[2].lower().replace('.', '')
                        if period in ['pm', 'pm'] and hour < 12:
                            hour += 12
                        elif period in ['am', 'am'] and hour == 12:
                            hour = 0
                        time_found = True
                    elif group_count == 2 and len(groups) > 1 and groups[1] in ['am', 'pm', 'a.m.', 'p.m.']:
                        hour = int(groups[0])
                        minute = 0
                        period = groups[1].lower().replace('.', '')
                        if period in ['pm', 'pm'] and hour < 12:
                            hour += 12
                        elif period in ['am', 'am'] and hour == 12:
                            hour = 0
                        time_found = True
                    break
                except (ValueError, AttributeError, IndexError):
                    pass
        
        # Time keywords
        if not time_found:
            time_keywords = {
                "noon": (12, 0), "lunch time": (12, 0),
                "morning": (8, 0), "evening": (18, 0),
                "night": (21, 0), "bedtime": (21, 0)
            }
            
            for keyword, (h, m) in time_keywords.items():
                if keyword in text_lower:
                    hour, minute = h, m
                    time_found = True
                    break
        
        # Build scheduled_time
        if time_found and hour is not None:
            base_date = now + timedelta(days=days_offset)
            scheduled_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if days_offset == 0 and scheduled_time < now:
                scheduled_time += timedelta(days=1)
        else:
            scheduled_time = now + timedelta(hours=1)
            scheduled_time = scheduled_time.replace(second=0, microsecond=0)
        
        return scheduled_time
    
    def _extract_recurrence(self, text_lower: str) -> Optional[str]:
        """Extract recurrence pattern."""
        if any(word in text_lower for word in ["every day", "daily", "everyday"]):
            return "daily"
        elif any(word in text_lower for word in ["every week", "weekly"]):
            return "weekly"
        elif any(word in text_lower for word in ["every month", "monthly"]):
            return "monthly"
        return None
    
    def _generate_title(self, text: str, category: str, entities: List[Dict]) -> str:
        """Generate a concise title from the text."""
        # Use category-based title
        category_titles = {
            "medication": "Take Medication",
            "appointment": "Appointment",
            "meal": "Meal Reminder",
            "hygiene": "Hygiene Reminder",
            "activity": "Activity Reminder",
            "general": "Reminder"
        }
        
        title = category_titles.get(category, "Reminder")
        
        # Try to extract specific medication name from entities
        if category == "medication" and entities:
            for entity in entities:
                if entity.get('entity_group') == 'MISC':
                    word = entity.get('word', '').strip()
                    if len(word) > 2:
                        title = f"Take {word.capitalize()}"
                        break
        
        return title
    
    def _extract_medication_names(self, text: str, entities: List[Dict]) -> List[str]:
        """Extract medication names from text and entities."""
        medications = []
        
        # From BERT entities
        for entity in entities:
            if entity.get('entity_group') in ['MISC', 'ORG']:
                word = entity.get('word', '').strip()
                if len(word) > 2 and word.lower() not in ['medication', 'medicine', 'pill']:
                    medications.append(word.capitalize())
        
        # Common medication patterns
        text_lower = text.lower()
        med_patterns = [
            r'(aspirin|ibuprofen|paracetamol|insulin|metformin)',
            r'(?:blue|red|white|yellow)\s+(?:pill|tablet)',
            r'blood\s+pressure\s+(?:medication|medicine|pill)'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, text_lower)
            medications.extend([m.capitalize() if isinstance(m, str) else m for m in matches])
        
        return list(set(medications))  # Remove duplicates


# Global instance
_bert_parser = None


def get_bert_parser() -> BERTReminderParser:
    """Get or create BERT parser instance."""
    global _bert_parser
    if _bert_parser is None:
        _bert_parser = BERTReminderParser()
    return _bert_parser
