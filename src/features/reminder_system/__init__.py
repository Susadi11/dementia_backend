"""
Context-Aware Smart Reminder System

This module implements an intelligent reminder system that:
- Uses BERT-based NLP to understand natural language commands
- Learns from user behavior patterns using Pitt Corpus trained models
- Adapts reminder schedules dynamically
- Escalates to caregivers when critical tasks are missed
"""

from .reminder_analyzer import PittBasedReminderAnalyzer
from .adaptive_scheduler import AdaptiveReminderScheduler
from .behavior_tracker import BehaviorTracker
from .reminder_models import (
    Reminder, ReminderInteraction, ReminderStatus, InteractionType, 
    ReminderPriority, BehaviorPattern, CaregiverAlert, ReminderCommand, ReminderResponse
)

__all__ = [
    'PittBasedReminderAnalyzer',
    'AdaptiveReminderScheduler',
    'BehaviorTracker',
    'Reminder',
    'ReminderInteraction',
    'ReminderStatus',
    'InteractionType',
    'ReminderPriority',
    'BehaviorPattern',
    'CaregiverAlert',
    'ReminderCommand',
    'ReminderResponse',
]
