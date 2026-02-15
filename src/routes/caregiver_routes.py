"""
Caregiver Routes
API endpoints for caregiver registration, authentication, and profile management
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
import logging
from ..services.caregiver_service import get_caregiver_service
from ..database import Database
from ..utils.auth import verify_token, refresh_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/caregiver", tags=["caregiver"])


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
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")
