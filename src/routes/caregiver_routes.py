"""
Caregiver Routes
API endpoints for caregiver registration, authentication, and profile management
"""

from fastapi import APIRouter, HTTPException, Depends, Header, Query
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from ..services.caregiver_service import get_caregiver_service
from ..services.reminder_db_service import ReminderDatabaseService
from ..database import Database
from ..utils.auth import verify_token, refresh_access_token
from ..features.reminder_system.behavior_tracker import BehaviorTracker
from ..features.reminder_system.weekly_report_generator import WeeklyReportGenerator
from ..features.reminder_system.reminder_analyzer import PittBasedReminderAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/caregiver", tags=["caregiver"])

# Initialize services
behavior_tracker = BehaviorTracker()
reminder_analyzer = PittBasedReminderAnalyzer()
report_generator = WeeklyReportGenerator(behavior_tracker)


# ===== REQUEST/RESPONSE MODELS =====

class CaregiverRegisterRequest(BaseModel):
    """Request model for caregiver registration"""
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    nic_number: str = Field(..., description="National Identity Card number")
    mobile_number: str = Field(..., description="Mobile phone number")
    district: str = Field(..., description="District or City")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    email: EmailStr = Field(..., description="Email address (used as username)")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    confirm_password: str = Field(..., description="Confirm password")
    profile_photo: Optional[str] = Field(None, description="Profile photo URL or base64")
    emergency_contact_name: str = Field(..., description="Emergency contact name")
    emergency_contact_number: str = Field(..., description="Emergency contact phone number")
    declaration_accepted: bool = Field(..., description="Declaration acceptance")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female', 'Other']
        if v not in allowed:
            raise ValueError(f'Gender must be one of: {", ".join(allowed)}')
        return v
    
    @validator('declaration_accepted')
    def validate_declaration(cls, v):
        if not v:
            raise ValueError('You must accept the declaration to register')
        return v


class CaregiverLoginRequest(BaseModel):
    """Request model for caregiver login"""
    email: EmailStr
    password: str


class CaregiverUpdateRequest(BaseModel):
    """Request model for updating caregiver profile"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    mobile_number: Optional[str] = None
    district: Optional[str] = None
    profile_photo: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    """Request model for changing password"""
    old_password: str
    new_password: str = Field(..., min_length=8)
    confirm_new_password: str
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v


class RefreshTokenRequest(BaseModel):
    """Request model for refreshing access token"""
    refresh_token: str


class LinkPatientRequest(BaseModel):
    """Request model for linking a patient to caregiver"""
    patient_id: str


class ProfilePhotoUploadRequest(BaseModel):
    """Request model for uploading profile photo"""
    photo_base64: str = Field(..., description="Base64 encoded image data")
    content_type: str = Field("image/jpeg", description="MIME type of the image")


# ===== DEPENDENCY FOR AUTH =====

async def get_current_caregiver(authorization: Optional[str] = Header(None)):
    """
    Dependency to get current authenticated caregiver from JWT token
    
    Usage: caregiver = Depends(get_current_caregiver)
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token, token_type="access")
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    if payload.get("role") != "caregiver":
        raise HTTPException(status_code=403, detail="Access forbidden: Not a caregiver account")
    
    return payload


# ===== ROUTES =====

@router.post("/register", response_model=dict)
async def register_caregiver(request: CaregiverRegisterRequest):
    """
    Register a new caregiver account
    
    This endpoint creates a new caregiver profile with a unique caregiver ID.
    The ID format is: C<Gender-Initial>-<FIRSTNAME>-<Last4Digits>
    Example: CF-SUSADI-1567
    """
    try:
        service = get_caregiver_service(Database.db)
        
        # Convert request to dict
        caregiver_data = request.dict()
        
        # Register caregiver
        result = await service.register_caregiver(caregiver_data)
        
        return {
            "success": True,
            "message": "Caregiver registered successfully",
            "caregiver": result
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=dict)
async def login_caregiver(request: CaregiverLoginRequest):
    """
    Caregiver login with email and password
    
    Returns JWT access token and refresh token for authenticated requests
    """
    try:
        service = get_caregiver_service(Database.db)
        
        # Authenticate caregiver
        result = await service.login_caregiver(request.email, request.password)
        
        return result
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/refresh-token", response_model=dict)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    
    When access token expires, use this endpoint with refresh token to get a new access token
    """
    try:
        new_access_token = refresh_access_token(request.refresh_token)
        
        if not new_access_token:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        return {
            "success": True,
            "access_token": new_access_token,
            "token_type": "Bearer"
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.get("/profile", response_model=dict)
async def get_profile(current_caregiver = Depends(get_current_caregiver)):
    """
    Get current caregiver's profile
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        profile = await service.get_caregiver_by_id(caregiver_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Caregiver not found")
        
        return {
            "success": True,
            "caregiver": profile
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.get("/profile/{caregiver_id}", response_model=dict)
async def get_caregiver_profile(
    caregiver_id: str,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get caregiver profile by ID
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        
        profile = await service.get_caregiver_by_id(caregiver_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Caregiver not found")
        
        return {
            "success": True,
            "caregiver": profile
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.put("/profile", response_model=dict)
async def update_profile(
    request: CaregiverUpdateRequest,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Update current caregiver's profile
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Get only non-None fields
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updated_profile = await service.update_caregiver_profile(caregiver_id, update_data)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "caregiver": updated_profile
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.post("/change-password", response_model=dict)
async def change_password(
    request: ChangePasswordRequest,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Change caregiver password
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        await service.change_password(
            caregiver_id,
            request.old_password,
            request.new_password
        )
        
        return {
            "success": True,
            "message": "Password changed successfully"
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Change password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.post("/link-patient", response_model=dict)
async def link_patient(
    request: LinkPatientRequest,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Link a patient to the current caregiver
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        updated_profile = await service.link_patient(caregiver_id, request.patient_id)
        
        return {
            "success": True,
            "message": f"Patient {request.patient_id} linked successfully",
            "caregiver": updated_profile
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Link patient error: {e}")
        raise HTTPException(status_code=500, detail="Failed to link patient")


@router.post("/unlink-patient", response_model=dict)
async def unlink_patient(
    request: LinkPatientRequest,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Unlink a patient from the current caregiver
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        updated_profile = await service.unlink_patient(caregiver_id, request.patient_id)
        
        return {
            "success": True,
            "message": f"Patient {request.patient_id} unlinked successfully",
            "caregiver": updated_profile
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unlink patient error: {e}")
        raise HTTPException(status_code=500, detail="Failed to unlink patient")


@router.get("/patients", response_model=dict)
async def get_linked_patients(current_caregiver = Depends(get_current_caregiver)):
    """
    Get list of patients linked to the current caregiver
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        profile = await service.get_caregiver_by_id(caregiver_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Caregiver not found")
        
        return {
            "success": True,
            "patient_ids": profile.get("patient_ids", []),
            "total_patients": len(profile.get("patient_ids", []))
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patients error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patients")


@router.delete("/profile", response_model=dict)
async def delete_profile(current_caregiver = Depends(get_current_caregiver)):
    """
    Delete current caregiver's account (soft delete)
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        await service.delete_caregiver(caregiver_id)
        
        return {
            "success": True,
            "message": "Account deleted successfully"
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")


@router.get("/lookup/{caregiver_id}", response_model=dict)
async def lookup_caregiver(caregiver_id: str):
    """
    Public endpoint to look up caregiver by ID.
    Returns limited info (name, phone, photo flag) for patient confirmation.
    No authentication required.
    """
    try:
        service = get_caregiver_service(Database.db)
        
        result = await service.lookup_caregiver(caregiver_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Caregiver not found")
        
        return {
            "success": True,
            "caregiver": result
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lookup caregiver error: {e}")
        raise HTTPException(status_code=500, detail="Failed to lookup caregiver")


@router.put("/profile-photo", response_model=dict)
async def upload_profile_photo(
    request: ProfilePhotoUploadRequest,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Upload caregiver profile photo as binary data stored in MongoDB.
    Accepts base64 encoded image data. Max 2MB.
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        result = await service.upload_profile_photo(
            caregiver_id, request.photo_base64, request.content_type
        )
        
        return result
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload photo error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload profile photo")


@router.get("/profile-photo/{caregiver_id}")
async def get_profile_photo(caregiver_id: str):
    """
    Get caregiver's profile photo as image response.
    Public endpoint so photos can be displayed anywhere.
    """
    try:
        service = get_caregiver_service(Database.db)
        
        result = await service.get_profile_photo(caregiver_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="No profile photo found")
        
        photo_bytes, content_type = result
        
        return Response(
            content=photo_bytes,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get photo error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile photo")


@router.get("/patients/details", response_model=dict)
async def get_patients_details(
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get full details of all patients linked to the current caregiver.
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        patients = await service.get_patients_details(caregiver_id)
        
        return {
            "success": True,
            "patients": patients,
            "total_patients": len(patients)
        }
        
    except RuntimeError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Get patients details error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patients details")


# ===== CAREGIVER DASHBOARD ENDPOINTS =====

@router.get("/dashboard/{patient_id}", response_model=dict)
async def get_patient_dashboard(
    patient_id: str,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get comprehensive dashboard data for a specific patient.
    
    Includes:
    - Reminder overview (completed, missed, pending)
    - Behavior pattern analysis using trained ML models
    - Cognitive risk assessment
    - Recent activity summary
    - Alerts and recommendations
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Initialize reminder database service
        reminder_db = ReminderDatabaseService()
        
        # Get all reminders for patient (last 30 days)
        from ..features.reminder_system.reminder_models import ReminderStatus
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)
        
        # Get current week reminders
        start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
        current_week_reminders = [
            r for r in all_reminders 
            if datetime.fromisoformat(r["scheduled_time"]) >= start_of_week
        ]
        
        # Calculate statistics
        total_reminders = len(current_week_reminders)
        completed = sum(1 for r in current_week_reminders if r["status"] == "completed")
        missed = sum(1 for r in current_week_reminders if r["status"] == "missed")
        pending = sum(1 for r in current_week_reminders if r["status"] == "active")
        
        completion_rate = (completed / total_reminders * 100) if total_reminders > 0 else 0
        
        # Get behavior pattern analysis using trained models
        behavior_pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=patient_id,
            days=30
        )
        
        # Get cognitive risk analysis
        risk_assessment = {
            "avg_cognitive_risk": behavior_pattern.avg_cognitive_risk_score or 0.0,
            "confusion_trend": behavior_pattern.confusion_trend,
            "escalation_recommended": behavior_pattern.escalation_recommended,
            "risk_level": _calculate_risk_level(behavior_pattern.avg_cognitive_risk_score)
        }
        
        # Get recent alerts
        alerts_collection = Database.get_collection("caregiver_alerts")
        recent_alerts_cursor = alerts_collection.find({
            "patient_id": patient_id,
            "caregiver_ids": caregiver_id,
            "created_at": {"$gte": datetime.now() - timedelta(days=7)}
        }).sort("created_at", -1).limit(10)
        
        recent_alerts = []
        async for alert in recent_alerts_cursor:
            alert["_id"] = str(alert["_id"])
            recent_alerts.append(alert)
        
        # Calculate week-over-week change
        last_week_start = start_of_week - timedelta(days=7)
        last_week_reminders = [
            r for r in all_reminders 
            if last_week_start <= datetime.fromisoformat(r["scheduled_time"]) < start_of_week
        ]
        last_week_completed = sum(1 for r in last_week_reminders if r["status"] == "completed")
        last_week_total = len(last_week_reminders)
        last_week_rate = (last_week_completed / last_week_total * 100) if last_week_total > 0 else 0
        week_change = completion_rate - last_week_rate
        
        return {
            "success": True,
            "patient_id": patient_id,
            "week_ending": datetime.now().strftime("%b %d, %Y"),
            "last_activity": _get_last_activity_time(current_week_reminders),
            "reminder_overview": {
                "compliance_rate": round(completion_rate, 0),
                "completed": completed,
                "missed": missed,
                "pending": pending,
                "total": total_reminders,
                "week_change": f"+{round(week_change, 0)}%" if week_change >= 0 else f"{round(week_change, 0)}%"
            },
            "behavior_analysis": {
                "avg_response_time_seconds": behavior_pattern.avg_response_time_seconds,
                "optimal_reminder_hour": behavior_pattern.optimal_reminder_hour,
                "worst_response_hours": behavior_pattern.worst_response_hours,
                "confirmed_count": behavior_pattern.confirmed_count,
                "ignored_count": behavior_pattern.ignored_count,
                "confused_count": behavior_pattern.confused_count,
                "delayed_count": behavior_pattern.delayed_count,
                "total_interactions": behavior_pattern.total_reminders
            },
            "cognitive_risk": risk_assessment,
            "recent_alerts": recent_alerts[:5],
            "total_alerts": len(recent_alerts),
            "recommendations": _generate_recommendations(behavior_pattern, risk_assessment)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patient dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve patient dashboard")


@router.get("/reminders/{patient_id}", response_model=dict)
async def get_patient_reminders(
    patient_id: str,
    status: Optional[str] = Query(None, description="Filter by status: active, completed, missed"),
    category: Optional[str] = Query(None, description="Filter by category: medication, appointment, meal, etc."),
    days: int = Query(7, description="Number of days to retrieve (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get all reminders for a specific patient with filtering options.
    
    Supports filtering by:
    - Status (active, completed, missed, snoozed)
    - Category (medication, appointment, meal, hygiene, etc.)
    - Time period (default: last 7 days)
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Initialize reminder database service
        reminder_db = ReminderDatabaseService()
        
        # Get reminders
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_reminders = [
            r for r in all_reminders
            if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]
        
        # Filter by status if provided
        if status:
            filtered_reminders = [r for r in filtered_reminders if r["status"] == status]
        
        # Filter by category if provided
        if category:
            filtered_reminders = [r for r in filtered_reminders if r.get("category") == category]
        
        # Sort by scheduled time (most recent first)
        filtered_reminders.sort(
            key=lambda x: datetime.fromisoformat(x["scheduled_time"]), 
            reverse=True
        )
        
        # Group by category for statistics
        category_stats = {}
        for reminder in filtered_reminders:
            cat = reminder.get("category", "general")
            if cat not in category_stats:
                category_stats[cat] = {
                    "total": 0,
                    "completed": 0,
                    "missed": 0,
                    "active": 0
                }
            category_stats[cat]["total"] += 1
            category_stats[cat][reminder["status"]] += 1
        
        return {
            "success": True,
            "patient_id": patient_id,
            "filters": {
                "status": status,
                "category": category,
                "days": days
            },
            "total_reminders": len(filtered_reminders),
            "reminders": filtered_reminders,
            "category_breakdown": category_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patient reminders error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve patient reminders")


@router.get("/reminders/{patient_id}/missed", response_model=dict)
async def get_missed_reminders(
    patient_id: str,
    days: int = Query(7, description="Number of days to check (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get all missed reminders for a patient this week.
    
    Returns detailed information about missed reminders including:
    - Reminder details (title, time, category)
    - Time when it was missed
    - Priority level
    - Associated risks
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Initialize reminder database service
        reminder_db = ReminderDatabaseService()
        
        # Get missed reminders
        from ..features.reminder_system.reminder_models import ReminderStatus
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        missed_reminders = [
            r for r in all_reminders
            if r["status"] == "missed" and datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]
        
        # Sort by scheduled time (most recent first)
        missed_reminders.sort(
            key=lambda x: datetime.fromisoformat(x["scheduled_time"]),
            reverse=True
        )
        
        # Format for frontend
        formatted_missed = []
        for reminder in missed_reminders:
            scheduled_time = datetime.fromisoformat(reminder["scheduled_time"])
            formatted_missed.append({
                "id": reminder["id"],
                "title": reminder["title"],
                "description": reminder.get("description"),
                "scheduled_time": scheduled_time.strftime("%b %d at %I:%M %p"),
                "scheduled_datetime": reminder["scheduled_time"],
                "category": reminder.get("category", "general"),
                "priority": reminder.get("priority", "medium"),
                "tag_color": _get_category_color(reminder.get("category", "general"))
            })
        
        return {
            "success": True,
            "patient_id": patient_id,
            "total_missed": len(missed_reminders),
            "missed_reminders": formatted_missed,
            "period_days": days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get missed reminders error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve missed reminders")


@router.get("/reminders/{patient_id}/snoozed", response_model=dict)
async def get_snoozed_reminders(
    patient_id: str,
    days: int = Query(7, description="Number of days to check (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get all snoozed reminders for a patient.

    Returns reminders that the patient acknowledged but postponed, including
    how many times they have been snoozed (snooze count).

    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]

        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")

        reminder_db = ReminderDatabaseService()
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)

        cutoff_date = datetime.now() - timedelta(days=days)
        snoozed_reminders = [
            r for r in all_reminders
            if r["status"] == "snoozed"
            and datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]

        snoozed_reminders.sort(
            key=lambda x: datetime.fromisoformat(x["scheduled_time"]),
            reverse=True
        )

        formatted_snoozed = []
        for reminder in snoozed_reminders:
            scheduled_time = datetime.fromisoformat(reminder["scheduled_time"])
            snoozed_until = reminder.get("snoozed_until")
            formatted_snoozed.append({
                "id": reminder["id"],
                "title": reminder["title"],
                "description": reminder.get("description"),
                "scheduled_time": scheduled_time.strftime("%b %d at %I:%M %p"),
                "scheduled_datetime": reminder["scheduled_time"],
                "snoozed_until": snoozed_until,
                "snooze_count": reminder.get("snooze_count", 1),
                "category": reminder.get("category", "general"),
                "priority": reminder.get("priority", "medium"),
                "tag_color": _get_category_color(reminder.get("category", "general"))
            })

        # Flag repeatedly snoozed reminders (snoozed 3+ times) as needing attention
        attention_needed = [r for r in formatted_snoozed if r["snooze_count"] >= 3]

        return {
            "success": True,
            "patient_id": patient_id,
            "total_snoozed": len(formatted_snoozed),
            "attention_needed_count": len(attention_needed),
            "snoozed_reminders": formatted_snoozed,
            "attention_needed": attention_needed,
            "period_days": days
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get snoozed reminders error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve snoozed reminders")


@router.get("/reminders/{patient_id}/grouped", response_model=dict)
async def get_reminders_grouped(
    patient_id: str,
    days: int = Query(7, description="Number of days to retrieve (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get all reminders for a patient grouped by status.

    Returns four groups for the caregiver dashboard reminder tab:
    - active   : pending / upcoming reminders
    - completed: successfully acknowledged
    - missed   : passed without response
    - snoozed  : postponed by the patient

    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]

        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")

        reminder_db = ReminderDatabaseService()
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)

        cutoff_date = datetime.now() - timedelta(days=days)
        period_reminders = [
            r for r in all_reminders
            if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]

        # Sort newest first
        period_reminders.sort(
            key=lambda x: datetime.fromisoformat(x["scheduled_time"]),
            reverse=True
        )

        def _fmt(r: dict) -> dict:
            t = datetime.fromisoformat(r["scheduled_time"])
            return {
                "id": r["id"],
                "title": r["title"],
                "description": r.get("description"),
                "scheduled_time": t.strftime("%b %d at %I:%M %p"),
                "scheduled_datetime": r["scheduled_time"],
                "category": r.get("category", "general"),
                "priority": r.get("priority", "medium"),
                "status": r["status"],
                "snooze_count": r.get("snooze_count", 0),
                "completed_at": r.get("completed_at"),
                "tag_color": _get_category_color(r.get("category", "general"))
            }

        grouped = {
            "active":    [_fmt(r) for r in period_reminders if r["status"] == "active"],
            "completed": [_fmt(r) for r in period_reminders if r["status"] == "completed"],
            "missed":    [_fmt(r) for r in period_reminders if r["status"] == "missed"],
            "snoozed":   [_fmt(r) for r in period_reminders if r["status"] == "snoozed"],
        }

        total = len(period_reminders)
        completed_count = len(grouped["completed"])
        compliance_rate = round((completed_count / total * 100), 1) if total > 0 else 0.0

        return {
            "success": True,
            "patient_id": patient_id,
            "period_days": days,
            "summary": {
                "total": total,
                "active": len(grouped["active"]),
                "completed": completed_count,
                "missed": len(grouped["missed"]),
                "snoozed": len(grouped["snoozed"]),
                "compliance_rate": compliance_rate
            },
            "reminders": grouped
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get grouped reminders error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve grouped reminders")


@router.get("/adherence-score/{patient_id}", response_model=dict)
async def get_adherence_and_risk_score(
    patient_id: str,
    days: int = Query(30, description="Analysis period in days (default: 30)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get a comprehensive adherence score and cognitive risk score for a patient.

    Uses trained ML models (BehaviorTracker + PittBasedReminderAnalyzer) to produce:
    - Adherence score  (0–100): how consistently the patient responds to reminders
    - Cognitive risk score (0–100): estimated cognitive risk level
    - Final combined wellness score (0–100)
    - Behavior trend: improving / stable / declining
    - Breakdown by reminder category (medication, meal, appointment, etc.)
    - Actionable flag counts (missed_critical, confusion_events, snoozed_repeatedly)

    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]

        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")

        # --- Reminder data ---
        reminder_db = ReminderDatabaseService()
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)

        cutoff_date = datetime.now() - timedelta(days=days)
        period_reminders = [
            r for r in all_reminders
            if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]

        total = len(period_reminders)
        completed = sum(1 for r in period_reminders if r["status"] == "completed")
        missed = sum(1 for r in period_reminders if r["status"] == "missed")
        snoozed = sum(1 for r in period_reminders if r["status"] == "snoozed")
        missed_critical = sum(
            1 for r in period_reminders
            if r["status"] == "missed" and r.get("priority") in ("high", "critical")
        )
        snoozed_repeatedly = sum(
            1 for r in period_reminders
            if r["status"] == "snoozed" and r.get("snooze_count", 0) >= 3
        )

        # Adherence score: weighted — completed full credit, snoozed partial, missed none
        # Formula: (completed*1.0 + snoozed*0.4) / total * 100
        adherence_score = round(
            ((completed * 1.0 + snoozed * 0.4) / total * 100) if total > 0 else 0.0, 1
        )

        # Category breakdown
        category_stats: Dict[str, Any] = {}
        for r in period_reminders:
            cat = r.get("category", "general")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "completed": 0, "missed": 0, "snoozed": 0, "adherence": 0.0}
            category_stats[cat]["total"] += 1
            category_stats[cat][r["status"]] += 1

        for cat, stats in category_stats.items():
            cat_total = stats["total"]
            cat_adherence = (
                (stats["completed"] * 1.0 + stats["snoozed"] * 0.4) / cat_total * 100
                if cat_total > 0 else 0.0
            )
            stats["adherence"] = round(cat_adherence, 1)

        # --- ML Behavior pattern ---
        behavior_pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=patient_id,
            days=days
        )

        raw_cognitive_risk = behavior_pattern.avg_cognitive_risk_score or 0.0
        cognitive_risk_score = round(raw_cognitive_risk * 100, 1)  # convert 0-1 → 0-100
        risk_level = _calculate_risk_level(raw_cognitive_risk)
        confusion_events = behavior_pattern.confused_count or 0

        # Behavior trend: compare first-half vs second-half of period
        half_days = days // 2
        cutoff_first_half = datetime.now() - timedelta(days=days)
        cutoff_second_half = datetime.now() - timedelta(days=half_days)

        first_half = [r for r in period_reminders if datetime.fromisoformat(r["scheduled_time"]) < cutoff_second_half]
        second_half = [r for r in period_reminders if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_second_half]

        def _compliance(reminders_list: list) -> float:
            t = len(reminders_list)
            c = sum(1 for r in reminders_list if r["status"] == "completed")
            return (c / t * 100) if t > 0 else 0.0

        first_rate = _compliance(first_half)
        second_rate = _compliance(second_half)
        delta = second_rate - first_rate

        if delta >= 5:
            trend = "improving"
        elif delta <= -5:
            trend = "declining"
        else:
            trend = "stable"

        # Final wellness score: higher adherence + lower cognitive risk = better
        # wellness = 0.6 * adherence_score + 0.4 * (100 - cognitive_risk_score)
        wellness_score = round(0.6 * adherence_score + 0.4 * (100.0 - cognitive_risk_score), 1)

        # Recommendation urgency
        if risk_level in ("critical", "high") or missed_critical > 2:
            urgency = "urgent"
        elif risk_level == "moderate" or trend == "declining":
            urgency = "monitor"
        else:
            urgency = "normal"

        return {
            "success": True,
            "patient_id": patient_id,
            "analysis_period_days": days,
            "scores": {
                "adherence_score": adherence_score,
                "cognitive_risk_score": cognitive_risk_score,
                "wellness_score": wellness_score,
                "risk_level": risk_level
            },
            "reminder_summary": {
                "total": total,
                "completed": completed,
                "missed": missed,
                "snoozed": snoozed,
                "missed_critical": missed_critical,
                "snoozed_repeatedly": snoozed_repeatedly,
                "confusion_events": confusion_events
            },
            "behavior_trend": {
                "trend": trend,
                "first_half_compliance": round(first_rate, 1),
                "second_half_compliance": round(second_rate, 1),
                "change_percent": round(delta, 1),
                "escalation_recommended": behavior_pattern.escalation_recommended
            },
            "category_breakdown": category_stats,
            "optimal_reminder_hour": behavior_pattern.optimal_reminder_hour,
            "urgency": urgency,
            "recommendations": _generate_recommendations(behavior_pattern, {
                "risk_level": risk_level,
                "escalation_recommended": behavior_pattern.escalation_recommended
            })
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get adherence score error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to calculate adherence score")


@router.get("/adherence-risk/{patient_id}", response_model=dict)
async def get_adherence_risk(
    patient_id: str,
    days: int = Query(30, description="Analysis period in days (default: 30)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get adherence risk assessment for a patient.

    Returns a risk-focused summary of the patient's adherence and cognitive state:
    - Overall risk level (low / moderate / high / critical)
    - Adherence rate and missed reminders
    - Cognitive risk score
    - Behavior trend
    - Actionable recommendations

    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]

        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")

        reminder_db = ReminderDatabaseService()
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)

        cutoff_date = datetime.now() - timedelta(days=days)
        period_reminders = [
            r for r in all_reminders
            if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]

        total = len(period_reminders)
        completed = sum(1 for r in period_reminders if r["status"] == "completed")
        missed = sum(1 for r in period_reminders if r["status"] == "missed")
        snoozed = sum(1 for r in period_reminders if r["status"] == "snoozed")
        missed_critical = sum(
            1 for r in period_reminders
            if r["status"] == "missed" and r.get("priority") in ("high", "critical")
        )

        adherence_rate = round(
            ((completed * 1.0 + snoozed * 0.4) / total * 100) if total > 0 else 0.0, 1
        )

        behavior_pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=patient_id,
            days=days
        )
        raw_cognitive_risk = behavior_pattern.avg_cognitive_risk_score or 0.0
        cognitive_risk_score = round(raw_cognitive_risk * 100, 1)
        risk_level = _calculate_risk_level(raw_cognitive_risk)

        # Trend
        half_days = days // 2
        first_half = [r for r in period_reminders if datetime.fromisoformat(r["scheduled_time"]) < datetime.now() - timedelta(days=half_days)]
        second_half = [r for r in period_reminders if datetime.fromisoformat(r["scheduled_time"]) >= datetime.now() - timedelta(days=half_days)]

        def _compliance(lst):
            t = len(lst)
            c = sum(1 for r in lst if r["status"] == "completed")
            return (c / t * 100) if t > 0 else 0.0

        delta = _compliance(second_half) - _compliance(first_half)
        trend = "improving" if delta >= 5 else ("declining" if delta <= -5 else "stable")

        return {
            "success": True,
            "patient_id": patient_id,
            "analysis_period_days": days,
            "risk_level": risk_level,
            "adherence_rate": adherence_rate,
            "cognitive_risk_score": cognitive_risk_score,
            "trend": trend,
            "summary": {
                "total": total,
                "completed": completed,
                "missed": missed,
                "snoozed": snoozed,
                "missed_critical": missed_critical,
                "confusion_events": behavior_pattern.confused_count or 0,
                "escalation_recommended": behavior_pattern.escalation_recommended
            },
            "recommendations": _generate_recommendations(behavior_pattern, {
                "risk_level": risk_level,
                "escalation_recommended": behavior_pattern.escalation_recommended
            })
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get adherence risk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to calculate adherence risk")


@router.get("/activity-completion/{patient_id}", response_model=dict)
async def get_activity_completion(
    patient_id: str,
    days: int = Query(7, description="Number of days to analyze (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get activity completion statistics for different reminder categories.
    
    Returns completion rates for:
    - Morning activities
    - Medication adherence
    - Hydration tracking
    - Social activities
    - Physical exercise
    - Other custom activities
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Initialize reminder database service
        reminder_db = ReminderDatabaseService()
        
        # Get all reminders
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reminders = [
            r for r in all_reminders
            if datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]
        
        # Group by category and calculate completion rates
        category_completion = {}
        activity_mapping = {
            "morning_routine": "Morning Walk",
            "hydration": "Hydration Check",
            "social": "Social Call",
            "reading": "Reading Time",
            "exercise": "Physical Exercise",
            "medication": "Medication",
            "meal": "Meal Reminder"
        }
        
        for category_key, activity_name in activity_mapping.items():
            category_reminders = [r for r in recent_reminders if r.get("category") == category_key or r.get("title", "").lower().find(activity_name.lower().split()[0]) != -1]
            
            if category_reminders:
                total = len(category_reminders)
                completed = sum(1 for r in category_reminders if r["status"] == "completed")
                completion_rate = round((completed / total * 100)) if total > 0 else 0
                
                category_completion[activity_name] = {
                    "completed": completed,
                    "total": total,
                    "completion_rate": completion_rate
                }
        
        # Format for frontend (match the screenshot structure)
        activities = []
        for activity_name, stats in category_completion.items():
            activities.append({
                "name": activity_name,
                "completion_rate": stats["completion_rate"],
                "completed": stats["completed"],
                "total": stats["total"]
            })
        
        return {
            "success": True,
            "patient_id": patient_id,
            "period_days": days,
            "activities": activities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get activity completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve activity completion")


@router.get("/medication-schedule/{patient_id}", response_model=dict)
async def get_medication_schedule(
    patient_id: str,
    days: int = Query(7, description="Number of days (default: 7)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get medication schedule and adherence tracking for a patient.
    
    Returns:
    - Daily medication schedule
    - Adherence percentage per medication
    - Missed doses
    - Upcoming medications
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Initialize reminder database service
        reminder_db = ReminderDatabaseService()
        
        # Get medication reminders
        all_reminders = await reminder_db.get_user_reminders(patient_id, limit=1000)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        medication_reminders = [
            r for r in all_reminders
            if r.get("category") == "medication" and datetime.fromisoformat(r["scheduled_time"]) >= cutoff_date
        ]
        
        # Group by medication name
        medications = {}
        for reminder in medication_reminders:
            med_name = reminder.get("title", "Unknown Medication")
            if med_name not in medications:
                medications[med_name] = {
                    "name": med_name,
                    "dosage": reminder.get("description", ""),
                    "times": [],
                    "schedule": {},
                    "total": 0,
                    "completed": 0,
                    "adherence": 0
                }
            
            medications[med_name]["total"] += 1
            if reminder["status"] == "completed":
                medications[med_name]["completed"] += 1
            
            # Track daily schedule
            scheduled_time = datetime.fromisoformat(reminder["scheduled_time"])
            day_name = scheduled_time.strftime("%a")  # Mon, Tue, etc.
            time_str = scheduled_time.strftime("%I:%M %p")
            
            if time_str not in medications[med_name]["times"]:
                medications[med_name]["times"].append(time_str)
            
            if day_name not in medications[med_name]["schedule"]:
                medications[med_name]["schedule"][day_name] = []
            
            medications[med_name]["schedule"][day_name].append({
                "time": time_str,
                "status": reminder["status"],
                "taken": reminder["status"] == "completed"
            })
        
        # Calculate adherence
        for med in medications.values():
            if med["total"] > 0:
                med["adherence"] = round((med["completed"] / med["total"]) * 100)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "period_days": days,
            "medications": list(medications.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get medication schedule error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve medication schedule")


@router.get("/behavior-analysis/{patient_id}", response_model=dict)
async def get_behavior_analysis(
    patient_id: str,
    days: int = Query(30, description="Analysis period in days (default: 30)"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get comprehensive behavior pattern analysis using trained ML models.
    
    Analyzes:
    - Response patterns and times
    - Cognitive risk scores from confusion detection model
    - Optimal reminder timing
    - Confusion trends
    - Recommended interventions
    
    Uses trained models:
    - Confusion detection model
    - Cognitive risk assessment model
    - Caregiver alert model
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Get behavior pattern using trained models
        behavior_pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=patient_id,
            days=days
        )
        
        # Calculate risk level based on cognitive risk score
        risk_level = _calculate_risk_level(behavior_pattern.avg_cognitive_risk_score)
        
        # Generate insights
        insights = []
        
        if behavior_pattern.optimal_reminder_hour:
            insights.append(f"Best response time is around {behavior_pattern.optimal_reminder_hour}:00")
        
        if behavior_pattern.confused_count > 0:
            confusion_rate = (behavior_pattern.confused_count / behavior_pattern.total_reminders * 100) if behavior_pattern.total_reminders > 0 else 0
            if confusion_rate > 10:
                insights.append(f"High confusion rate detected ({round(confusion_rate)}%)")
        
        if behavior_pattern.ignored_count > behavior_pattern.confirmed_count:
            insights.append("Patient often ignores reminders - consider increasing urgency")
        
        if behavior_pattern.escalation_recommended:
            insights.append("Escalation recommended - consider medical consultation")
        
        return {
            "success": True,
            "patient_id": patient_id,
            "analysis_period_days": days,
            "behavior_summary": {
                "total_reminders": behavior_pattern.total_reminders,
                "confirmed_count": behavior_pattern.confirmed_count,
                "ignored_count": behavior_pattern.ignored_count,
                "confused_count": behavior_pattern.confused_count,
                "delayed_count": behavior_pattern.delayed_count,
                "avg_response_time_seconds": behavior_pattern.avg_response_time_seconds
            },
            "cognitive_assessment": {
                "avg_risk_score": round(behavior_pattern.avg_cognitive_risk_score or 0, 2),
                "risk_level": risk_level,
                "confusion_trend": behavior_pattern.confusion_trend,
                "escalation_recommended": behavior_pattern.escalation_recommended
            },
            "timing_analysis": {
                "optimal_hour": behavior_pattern.optimal_reminder_hour,
                "worst_hours": behavior_pattern.worst_response_hours,
                "recommended_time_adjustment_minutes": behavior_pattern.recommended_time_adjustment_minutes
            },
            "recommendations": {
                "frequency_multiplier": behavior_pattern.recommended_frequency_multiplier,
                "time_adjustment": behavior_pattern.recommended_time_adjustment_minutes,
                "escalation_needed": behavior_pattern.escalation_recommended
            },
            "insights": insights,
            "last_updated": behavior_pattern.last_updated.isoformat() if behavior_pattern.last_updated else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get behavior analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve behavior analysis")


@router.get("/weekly-report/{patient_id}", response_model=dict)
async def get_weekly_report(
    patient_id: str,
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Generate comprehensive weekly cognitive and behavior report.
    
    Includes:
    - Reminder completion statistics
    - Daily risk summaries
    - Cognitive health metrics
    - Caregiver alerts
    - Week-over-week comparisons
    - Actionable recommendations
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Parse end date or use current date
        if end_date:
            report_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            report_end_date = datetime.now()
        
        # Generate weekly report using trained models
        weekly_report = report_generator.generate_weekly_report(
            user_id=patient_id,
            end_date=report_end_date,
            caregiver_ids=[caregiver_id]
        )
        
        return {
            "success": True,
            "patient_id": patient_id,
            "report": weekly_report.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get weekly report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate weekly report")


@router.get("/alerts/{patient_id}", response_model=dict)
async def get_caregiver_alerts(
    patient_id: str,
    days: int = Query(7, description="Number of days (default: 7)"),
    priority: Optional[str] = Query(None, description="Filter by priority: low, medium, high, critical"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Get caregiver alerts for a specific patient.
    
    Returns alerts generated by the trained caregiver alert model based on:
    - High cognitive risk scores
    - Repeated confusion
    - Missed critical reminders
    - Unusual behavior patterns
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_caregiver_service(Database.db)
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Verify caregiver has access to this patient
        profile = await service.get_caregiver_by_id(caregiver_id)
        if patient_id not in profile.get("patient_ids", []):
            raise HTTPException(status_code=403, detail="Access denied to this patient")
        
        # Get alerts from database
        alerts_collection = Database.get_collection("caregiver_alerts")
        
        # Build query
        query = {
            "patient_id": patient_id,
            "caregiver_ids": caregiver_id
        }
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query["created_at"] = {"$gte": cutoff_date}
        
        if priority:
            query["priority"] = priority
        
        if resolved is not None:
            query["resolved"] = resolved
        
        # Fetch alerts
        cursor = alerts_collection.find(query).sort("created_at", -1).limit(100)
        
        alerts = []
        async for alert in cursor:
            alert["_id"] = str(alert["_id"])
            alerts.append(alert)
        
        # Count by priority
        priority_counts = {
            "critical": sum(1 for a in alerts if a.get("priority") == "critical"),
            "high": sum(1 for a in alerts if a.get("priority") == "high"),
            "medium": sum(1 for a in alerts if a.get("priority") == "medium"),
            "low": sum(1 for a in alerts if a.get("priority") == "low")
        }
        
        unresolved_count = sum(1 for a in alerts if not a.get("resolved", False))
        
        return {
            "success": True,
            "patient_id": patient_id,
            "total_alerts": len(alerts),
            "unresolved_alerts": unresolved_count,
            "priority_breakdown": priority_counts,
            "alerts": alerts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get caregiver alerts error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve caregiver alerts")


@router.post("/alerts/{alert_id}/resolve", response_model=dict)
async def resolve_alert(
    alert_id: str,
    current_caregiver = Depends(get_current_caregiver)
):
    """
    Mark a caregiver alert as resolved.
    
    Requires: Bearer token in Authorization header
    """
    try:
        caregiver_id = current_caregiver["caregiver_id"]
        
        # Update alert
        alerts_collection = Database.get_collection("caregiver_alerts")
        
        result = await alerts_collection.update_one(
            {"_id": alert_id, "caregiver_ids": caregiver_id},
            {
                "$set": {
                    "resolved": True,
                    "resolved_at": datetime.now(),
                    "resolved_by": caregiver_id
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Alert not found or access denied")
        
        return {
            "success": True,
            "message": "Alert resolved successfully",
            "alert_id": alert_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolve alert error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


# ===== HELPER FUNCTIONS =====

def _calculate_risk_level(cognitive_risk_score: Optional[float]) -> str:
    """Calculate risk level category from cognitive risk score."""
    if cognitive_risk_score is None:
        return "unknown"
    
    if cognitive_risk_score >= 0.75:
        return "critical"
    elif cognitive_risk_score >= 0.50:
        return "high"
    elif cognitive_risk_score >= 0.25:
        return "moderate"
    else:
        return "low"


def _get_last_activity_time(reminders: List[Dict[str, Any]]) -> str:
    """Get the last activity time in human-readable format."""
    if not reminders:
        return "No recent activity"
    
    # Find most recent completed reminder
    completed_reminders = [r for r in reminders if r["status"] == "completed" and r.get("completed_at")]
    
    if not completed_reminders:
        return "No completed activities"
    
    latest = max(completed_reminders, key=lambda x: datetime.fromisoformat(x["completed_at"]))
    completed_time = datetime.fromisoformat(latest["completed_at"])
    
    # Calculate time difference
    now = datetime.now()
    diff = now - completed_time
    
    if diff.total_seconds() < 3600:  # Less than 1 hour
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff.total_seconds() < 86400:  # Less than 1 day
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"


def _get_category_color(category: str) -> str:
    """Get color tag for reminder category."""
    color_map = {
        "medication": "Medication",
        "appointment": "Activity",
        "meal": "Activity",
        "exercise": "Activity",
        "social": "Activity",
        "hygiene": "Activity",
        "morning_routine": "Activity",
        "hydration": "Activity"
    }
    return color_map.get(category, "Activity")


def _generate_recommendations(behavior_pattern: Any, risk_assessment: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on behavior analysis."""
    recommendations = []
    
    if risk_assessment["risk_level"] in ["high", "critical"]:
        recommendations.append("Schedule medical consultation - elevated cognitive risk detected")
    
    if behavior_pattern.escalation_recommended:
        recommendations.append("Consider increasing caregiver involvement")
    
    if behavior_pattern.ignored_count > behavior_pattern.confirmed_count:
        recommendations.append("Adjust reminder delivery method or timing")
    
    if behavior_pattern.confused_count > 5:
        recommendations.append("Simplify reminder messages and reduce complexity")
    
    if behavior_pattern.optimal_reminder_hour:
        recommendations.append(f"Schedule important reminders around {behavior_pattern.optimal_reminder_hour}:00")
    
    if not recommendations:
        recommendations.append("Continue current reminder schedule - patient responding well")
    
    return recommendations
