"""
User Routes
API endpoints for user (patient/elderly) registration, authentication, and profile management
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
import logging
from ..services.user_service import get_user_service
from ..database import Database
from ..utils.auth import verify_token, refresh_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["user"])


# ===== REQUEST/RESPONSE MODELS =====

class UserRegisterRequest(BaseModel):
    """Request model for user registration"""
    full_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr = Field(..., description="Email address (used as username)")
    phone_number: Optional[str] = Field(None, description="Phone number")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")
    gender: Optional[str] = Field(None, description="Gender: Male, Female, or Other")
    address: Optional[str] = Field(None, description="Home address")
    emergency_contact_name: Optional[str] = Field(None, description="Emergency contact name")
    emergency_contact_number: Optional[str] = Field(None, description="Emergency contact phone")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    confirm_password: str = Field(..., description="Confirm password")
    profile_photo: Optional[str] = Field(None, description="Profile photo URL or base64")
    medical_conditions: Optional[List[str]] = Field(default=[], description="List of medical conditions")
    caregiver_id: Optional[str] = Field(None, description="Linked caregiver ID")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        if v is not None:
            allowed = ['Male', 'Female', 'Other']
            if v not in allowed:
                raise ValueError(f'Gender must be one of: {", ".join(allowed)}')
        return v


class UserLoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str


class UserUpdateRequest(BaseModel):
    """Request model for updating user profile"""
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None
    address: Optional[str] = None
    profile_photo: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_number: Optional[str] = None
    medical_conditions: Optional[List[str]] = None
    caregiver_id: Optional[str] = None


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


class GoogleLoginRequest(BaseModel):
    """Request model for Google Sign-In"""
    id_token: str = Field(..., description="Google ID token from client")


class RefreshTokenRequest(BaseModel):
    """Request model for refreshing access token"""
    refresh_token: str


class LinkCaregiverRequest(BaseModel):
    """Request model for linking a caregiver to user"""
    caregiver_id: str


class ForgotPasswordRequest(BaseModel):
    """Request model for forgot password"""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Request model for reset password"""
    email: EmailStr
    reset_code: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v



# ===== DEPENDENCY FOR AUTH =====

async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Dependency to get current authenticated user from JWT token
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")
        payload = verify_token(token)
        
        if payload.get("role") != "user":
            raise HTTPException(status_code=403, detail="Not authorized as user")
        
        return payload
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ===== ROUTES =====

@router.post("/register", response_model=dict)
async def register_user(request: UserRegisterRequest):
    """
    Register a new user account
    
    Returns user ID and profile information
    """
    try:
        service = get_user_service()
        
        user = await service.register_user(
            full_name=request.full_name,
            email=request.email,
            password=request.password,
            phone_number=request.phone_number,
            age=request.age,
            gender=request.gender,
            address=request.address,
            emergency_contact_name=request.emergency_contact_name,
            emergency_contact_number=request.emergency_contact_number,
            profile_photo=request.profile_photo,
            medical_conditions=request.medical_conditions,
            caregiver_id=request.caregiver_id
        )
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": user
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")


@router.post("/login", response_model=dict)
async def login_user(request: UserLoginRequest):
    """
    Login user and return JWT tokens
    
    Returns access token, refresh token, and user profile
    """
    try:
        service = get_user_service()
        
        result = await service.login_user(request.email, request.password)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/google-login", response_model=dict)
async def google_login(request: GoogleLoginRequest):
    """
    Login or register user via Google Sign-In.

    Verifies the Google ID token and returns JWT tokens.
    Creates a new user account if the email is not registered.
    """
    try:
        service = get_user_service()
        result = await service.google_login(request.id_token)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Google login error: {e}")
        raise HTTPException(status_code=500, detail="Google login failed")


@router.post("/refresh-token", response_model=dict)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    
    Returns new access token
    """
    try:
        new_access_token = refresh_access_token(request.refresh_token)
        
        return {
            "success": True,
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")


@router.get("/profile", response_model=dict)
async def get_profile(current_user = Depends(get_current_user)):
    """
    Get current user's profile
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        user_id = current_user["user_id"]
        
        profile = await service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.get("/profile/{user_id}", response_model=dict)
async def get_user_profile(
    user_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get user profile by ID
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        
        profile = await service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.put("/profile", response_model=dict)
async def update_profile(
    request: UserUpdateRequest,
    current_user = Depends(get_current_user)
):
    """
    Update current user's profile
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        user_id = current_user["user_id"]
        
        # Get only non-None fields
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updated_profile = await service.update_user_profile(user_id, update_data)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "user": updated_profile
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.post("/change-password", response_model=dict)
async def change_password(
    request: ChangePasswordRequest,
    current_user = Depends(get_current_user)
):
    """
    Change current user's password
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        user_id = current_user["user_id"]
        
        await service.change_password(
            user_id,
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


@router.post("/link-caregiver", response_model=dict)
async def link_caregiver(
    request: LinkCaregiverRequest,
    current_user = Depends(get_current_user)
):
    """
    Link a caregiver to the current user
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        user_id = current_user["user_id"]
        
        updated_profile = await service.link_to_caregiver(user_id, request.caregiver_id)
        
        return {
            "success": True,
            "message": f"Caregiver {request.caregiver_id} linked successfully",
            "user": updated_profile
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Link caregiver error: {e}")
        raise HTTPException(status_code=500, detail="Failed to link caregiver")


@router.delete("/profile", response_model=dict)
async def delete_profile(current_user = Depends(get_current_user)):
    """
    Delete current user's account (soft delete)
    
    Requires: Bearer token in Authorization header
    """
    try:
        service = get_user_service()
        user_id = current_user["user_id"]
        
        await service.delete_user(user_id)
        
        return {
            "success": True,
            "message": "Account deleted successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")


@router.post("/forgot-password", response_model=dict)
async def forgot_password(request: ForgotPasswordRequest):
    """
    Request password reset code
    
    Generates a 6-digit reset code and stores it temporarily.
    In production, this code should be sent via email.
    For testing, the code is returned in the response.
    """
    try:
        service = get_user_service()
        
        reset_code = await service.generate_password_reset_code(request.email)
        
        return {
            "success": True,
            "message": "Password reset code generated",
            "email": request.email,
            "reset_code": reset_code  # In production, don't return this - send via email
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process forgot password request")


@router.post("/reset-password", response_model=dict)
async def reset_password(request: ResetPasswordRequest):
    """
    Reset password using reset code
    
    Verifies the reset code and updates the password
    """
    try:
        service = get_user_service()
        
        await service.reset_password_with_code(
            request.email,
            request.reset_code,
            request.new_password
        )
        
        return {
            "success": True,
            "message": "Password reset successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")

