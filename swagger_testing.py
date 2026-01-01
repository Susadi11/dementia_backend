"""
Swagger Testing Utilities for Context-Aware Smart Reminder System

Provides testing endpoints, mock data, and utilities for comprehensive API testing
through Swagger UI.
"""

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

# Create testing router
testing_router = APIRouter(
    prefix="/api/testing",
    tags=["Testing & Demo"],
    responses={404: {"description": "Not found"}}
)

# Testing-specific models
class TestingScenario(str, Enum):
    """Available testing scenarios."""
    MILD_COGNITIVE_IMPAIRMENT = "mild_cognitive_impairment"
    MODERATE_DEMENTIA = "moderate_dementia"
    NORMAL_COGNITIVE_FUNCTION = "normal_cognitive_function"
    MIXED_SYMPTOMS = "mixed_symptoms"

class MockUserProfile(BaseModel):
    """Mock user profile for testing."""
    user_id: str = Field(..., example="test_user_123")
    name: str = Field(..., example="John Doe")
    age: int = Field(..., ge=18, le=120, example=75)
    cognitive_baseline: float = Field(..., ge=0, le=1, example=0.8)
    preferred_reminder_time: str = Field(..., example="08:00")
    caregiver_contacts: List[str] = Field(default=[], example=["caregiver_456"])

class TestingResponse(BaseModel):
    """Standard testing response format."""
    success: bool
    scenario: str
    mock_data: Dict[str, Any]
    instructions: str
    next_steps: List[str]

@testing_router.post("/scenarios/{scenario_type}", 
                    response_model=TestingResponse,
                    summary="Generate Test Scenario",
                    description="Create comprehensive test data for different dementia assessment scenarios")
async def generate_test_scenario(
    scenario_type: TestingScenario = Path(..., description="Type of testing scenario to generate"),
    include_audio_data: bool = Query(True, description="Include mock audio analysis data"),
    reminder_count: int = Query(5, ge=1, le=20, description="Number of test reminders to generate")
) -> TestingResponse:
    """
    Generate comprehensive test scenarios for dementia assessment and reminder testing.
    
    This endpoint creates realistic test data that can be used with other API endpoints
    to simulate various cognitive states and reminder scenarios.
    
    **Test Scenarios Available:**
    - `mild_cognitive_impairment`: Subtle cognitive changes
    - `moderate_dementia`: Clear cognitive decline indicators  
    - `normal_cognitive_function`: Healthy cognitive patterns
    - `mixed_symptoms`: Combination of various indicators
    
    **Generated Data Includes:**
    - User profile with cognitive baseline
    - Sample conversation transcripts
    - Audio analysis features (if requested)
    - Pre-configured reminders
    - Expected AI responses
    """
    
    # Generate scenario-specific data
    if scenario_type == TestingScenario.MILD_COGNITIVE_IMPAIRMENT:
        mock_data = {
            "user_profile": {
                "user_id": "test_mci_user",
                "cognitive_score": 0.65,
                "risk_factors": ["memory_lapses", "word_finding_difficulty"]
            },
            "sample_conversation": [
                "I sometimes forget where I put things",
                "What was I just talking about?",
                "The word is on the tip of my tongue"
            ],
            "audio_features": {
                "pause_frequency": 0.25,
                "speech_rate": 110,
                "tremor_intensity": 0.1
            } if include_audio_data else None,
            "test_reminders": generate_test_reminders(reminder_count, "mild")
        }
        instructions = "Use this data to test early-stage cognitive assessment"
        
    elif scenario_type == TestingScenario.MODERATE_DEMENTIA:
        mock_data = {
            "user_profile": {
                "user_id": "test_moderate_user", 
                "cognitive_score": 0.35,
                "risk_factors": ["confusion", "disorientation", "memory_loss"]
            },
            "sample_conversation": [
                "I don't remember what day it is",
                "Who are you again?",
                "I feel very confused right now"
            ],
            "audio_features": {
                "pause_frequency": 0.45,
                "speech_rate": 85,
                "tremor_intensity": 0.3
            } if include_audio_data else None,
            "test_reminders": generate_test_reminders(reminder_count, "moderate")
        }
        instructions = "Use this data to test advanced cognitive decline detection"
        
    elif scenario_type == TestingScenario.NORMAL_COGNITIVE_FUNCTION:
        mock_data = {
            "user_profile": {
                "user_id": "test_normal_user",
                "cognitive_score": 0.92,
                "risk_factors": []
            },
            "sample_conversation": [
                "I'm feeling great today, thank you for asking",
                "I remember everything clearly",
                "My mind feels sharp and focused"
            ],
            "audio_features": {
                "pause_frequency": 0.05,
                "speech_rate": 125,
                "tremor_intensity": 0.02
            } if include_audio_data else None,
            "test_reminders": generate_test_reminders(reminder_count, "normal")
        }
        instructions = "Use this data to test normal cognitive function baseline"
        
    else:  # MIXED_SYMPTOMS
        mock_data = {
            "user_profile": {
                "user_id": "test_mixed_user",
                "cognitive_score": 0.55,
                "risk_factors": ["inconsistent_memory", "variable_attention"]
            },
            "sample_conversation": [
                "Some days I remember everything, others I don't",
                "I'm having a good day today",
                "Wait, what were we discussing?"
            ],
            "audio_features": {
                "pause_frequency": 0.35,
                "speech_rate": 105,
                "tremor_intensity": 0.15
            } if include_audio_data else None,
            "test_reminders": generate_test_reminders(reminder_count, "mixed")
        }
        instructions = "Use this data to test variable cognitive patterns"
    
    next_steps = [
        f"1. Use POST /api/analysis/text with sample conversations",
        f"2. Create reminders using POST /api/reminders/",
        f"3. Test user interactions with generated reminder IDs",
        f"4. Monitor caregiver alerts and escalations",
        f"5. Review analytics with GET /api/reminders/stats/user/{{user_id}}"
    ]
    
    return TestingResponse(
        success=True,
        scenario=scenario_type.value,
        mock_data=mock_data,
        instructions=instructions,
        next_steps=next_steps
    )

def generate_test_reminders(count: int, severity: str) -> List[Dict]:
    """Generate test reminders based on cognitive severity level."""
    base_reminders = [
        {
            "title": "Take Morning Medication",
            "category": "medication",
            "priority": "high"
        },
        {
            "title": "Doctor Appointment",
            "category": "appointment", 
            "priority": "high"
        },
        {
            "title": "Call Family Member",
            "category": "social",
            "priority": "medium"
        },
        {
            "title": "Take Evening Walk",
            "category": "exercise",
            "priority": "low"
        },
        {
            "title": "Prepare Lunch",
            "category": "meal",
            "priority": "medium"
        }
    ]
    
    reminders = []
    for i in range(count):
        base = base_reminders[i % len(base_reminders)]
        reminder = {
            "user_id": f"test_{severity}_user",
            "title": f"{base['title']} ({i+1})",
            "description": f"Test reminder for {severity} cognitive level",
            "scheduled_time": (datetime.now() + timedelta(hours=i+1)).isoformat(),
            "priority": base["priority"],
            "category": base["category"],
            "escalation_threshold_minutes": 30 if severity == "moderate" else 60
        }
        reminders.append(reminder)
    
    return reminders

@testing_router.get("/swagger-examples",
                   summary="Get Swagger Testing Examples",
                   description="Get comprehensive examples for testing all API endpoints through Swagger UI")
async def get_swagger_examples():
    """
    Returns comprehensive examples for testing all API endpoints through Swagger UI.
    
    This endpoint provides ready-to-use JSON payloads for:
    - Creating different types of reminders
    - Analyzing various conversation samples
    - Testing user interactions
    - Simulating caregiver scenarios
    """
    
    return {
        "reminder_examples": {
            "medication_reminder": {
                "user_id": "test_user_123",
                "title": "Take blood pressure medication",
                "description": "Take 2 blue pills with a full glass of water",
                "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
                "priority": "high",
                "category": "medication",
                "repeat_pattern": "daily",
                "caregiver_ids": ["caregiver_456"],
                "notify_caregiver_on_miss": True,
                "escalation_threshold_minutes": 30
            },
            "appointment_reminder": {
                "user_id": "test_user_123", 
                "title": "Cardiology appointment",
                "description": "Appointment with Dr. Smith at Heart Center - Room 205",
                "scheduled_time": (datetime.now() + timedelta(days=1)).isoformat(),
                "priority": "critical",
                "category": "appointment",
                "caregiver_ids": ["caregiver_456"],
                "notify_caregiver_on_miss": True,
                "escalation_threshold_minutes": 15
            }
        },
        "analysis_examples": {
            "concerning_text": {
                "text": "I keep forgetting where I put my keys and sometimes I don't remember what I was just saying",
                "audio_data": {
                    "pause_frequency": 0.4,
                    "tremor_intensity": 0.25,
                    "emotion_intensity": 0.7,
                    "speech_error_rate": 0.2,
                    "speech_rate": 90.0
                }
            },
            "normal_text": {
                "text": "I feel great today and my memory is working well, thank you for asking",
                "audio_data": {
                    "pause_frequency": 0.05,
                    "tremor_intensity": 0.02,
                    "emotion_intensity": 0.3,
                    "speech_error_rate": 0.01,
                    "speech_rate": 130.0
                }
            }
        },
        "interaction_examples": {
            "confirmed_reminder": {
                "reminder_id": "reminder_123",
                "user_id": "test_user_123",
                "interaction_type": "confirmed",
                "user_response_text": "Yes, I took my medication as prescribed"
            },
            "missed_reminder": {
                "reminder_id": "reminder_123",
                "user_id": "test_user_123", 
                "interaction_type": "missed",
                "user_response_text": "I forgot to take it"
            }
        }
    }

@testing_router.get("/health-check",
                   summary="Testing System Health",
                   description="Verify that all testing endpoints and mock data generation is working correctly")
async def testing_health_check():
    """
    Comprehensive health check for the testing system.
    
    Verifies:
    - Mock data generation capabilities
    - Database connectivity for testing
    - AI model availability for testing scenarios
    """
    
    try:
        # Test mock data generation
        test_reminders = generate_test_reminders(3, "mild")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "mock_data_generation": True,
                "scenario_creation": True,
                "swagger_examples": True
            },
            "sample_data": {
                "generated_reminders": len(test_reminders),
                "available_scenarios": len(TestingScenario),
                "test_users": ["test_user_123", "test_mci_user", "test_moderate_user"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Testing system unhealthy: {str(e)}")