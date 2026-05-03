"""
Crisis Detector

Detects suicidal ideation and crisis language in patient chat messages.
Triggers immediate caregiver alert via WebSocket + MongoDB when detected.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Crisis keyword/phrase patterns — ordered most-specific first
CRISIS_PATTERNS = [
    r"\bsuicid(e|al|ally)\b",
    r"\bkill\s+my\s*self\b",
    r"\bend\s+my\s+(life|pain|suffering)\b",
    r"\bwant\s+to\s+die\b",
    r"\bneed\s+to\s+die\b",
    r"\bi\s+should\s+(be\s+dead|die)\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist|be\s+here)\b",
    r"\bnot\s+want\s+to\s+(live|be\s+alive|exist)\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bbetter\s+off\s+without\s+me\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on|keep\s+going)\b",
    r"\b(can'?t|cannot)\s+(go\s+on|take\s+it\s+anymore|live\s+like\s+this)\b",
    r"\bwant\s+to\s+disappear\b",
    r"\bself.?harm\b",
    r"\bhurt\s+my\s*self\b",
    r"\bi\s+can'?t\s+be\s+here\b",
    r"\bdon'?t\s+want\s+to\s+be\s+here\b",
    r"\bi\s+cant\s+be\s+here\b",
    r"\bno\s+point\s+(in\s+living|living|to\s+live)\b",
    r"\bready\s+to\s+die\b",
    r"\bwish\s+i\s+(was|were|am)\s+(dead|gone|never\s+born)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]


def detect_crisis(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scan message text for crisis / suicidal language.

    Args:
        text: Patient's message text

    Returns:
        (is_crisis, matched_phrase) — matched_phrase is None if no crisis detected
    """
    if not text or not text.strip():
        return False, None

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            matched = match.group(0)
            logger.warning(f"[CRISIS] Crisis phrase detected: '{matched}'")
            return True, matched

    return False, None
