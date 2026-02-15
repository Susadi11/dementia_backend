"""
Authentication Utilities
Handles JWT token generation, validation, and password hashing
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
import bcrypt
import os
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    Args:
        password: Plain text password
        hashed_password: Hashed password to compare against
    
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token (e.g., user_id, email, role)
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create JWT refresh token
    
    Args:
        data: Data to encode in token (usually just user_id)
    
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        jwt.ExpiredSignatureError: If token has expired
        jwt.InvalidTokenError: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verify JWT token and check its type
    
    Args:
        token: JWT token string
        token_type: Expected token type ("access" or "refresh")
    
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = decode_token(token)
        
        # Check token type
        if payload.get("type") != token_type:
            logger.warning(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
            return None
        
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def create_tokens_for_caregiver(caregiver: Dict[str, Any]) -> Dict[str, str]:
    """
    Create access and refresh tokens for a caregiver
    
    Args:
        caregiver: Caregiver data dictionary
    
    Returns:
        Dictionary with access_token and refresh_token
    """
    token_data = {
        "caregiver_id": caregiver["caregiver_id"],
        "email": caregiver["email"],
        "role": "caregiver",
        "full_name": caregiver.get("full_name", "")
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"caregiver_id": caregiver["caregiver_id"]})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }


def create_tokens_for_patient(patient: Dict[str, Any]) -> Dict[str, str]:
    """
    Create access and refresh tokens for a patient
    
    Args:
        patient: Patient data dictionary
    
    Returns:
        Dictionary with access_token and refresh_token
    """
    token_data = {
        "patient_id": patient.get("patient_id") or patient.get("_id"),
        "email": patient.get("email", ""),
        "role": "patient",
        "name": patient.get("name", "")
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"patient_id": patient.get("patient_id") or patient.get("_id")})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Generate new access token from refresh token
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        New access token if refresh token is valid, None otherwise
    """
    payload = verify_token(refresh_token, token_type="refresh")
    
    if not payload:
        return None
    
    # Create new access token with limited data
    new_token_data = {
        k: v for k, v in payload.items() 
        if k not in ["exp", "iat", "type"]
    }
    
    new_access_token = create_access_token(new_token_data)
    return new_access_token
