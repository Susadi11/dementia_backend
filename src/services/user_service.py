"""
User Service
Handles business logic for user (patient/elderly) registration, authentication, and profile management
"""

from typing import Dict, Any, Optional
import logging
import hashlib
import random
import os
import base64
from datetime import datetime, timedelta
from bson import Binary
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from ..utils.auth import hash_password, verify_password, create_access_token, create_refresh_token
from ..database import Database

# Use Web Client ID for verifying tokens from Android app
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "717354778892-cejp1dj272oc05qrjtu6a9bhcfb5lq30.apps.googleusercontent.com")

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing user/patient accounts"""
    
    def __init__(self):
        self.db = Database.db
        self.collection_name = "users"
        logger.info("User service initialized")
    
    async def _get_collection(self):
        """Get or create users collection"""
        if self.db is None:
            raise ValueError("Database connection not initialized")
        return self.db[self.collection_name]
    
    def generate_user_id(self, full_name: str, age: int) -> str:
        """
        Generate unique user ID
        Format: USER-JOHN-45-1234
        """
        # Extract first name
        name_parts = full_name.strip().split()
        first_name = name_parts[0].upper() if name_parts else "USER"
        
        # Generate hash from timestamp
        timestamp_hash = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:4].upper()
        
        user_id = f"USER-{first_name}-{age}-{timestamp_hash}"
        return user_id
    
    async def register_user(
        self,
        full_name: str,
        email: str,
        password: str,
        phone_number: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        address: Optional[str] = None,
        emergency_contact_name: Optional[str] = None,
        emergency_contact_number: Optional[str] = None,
        profile_photo: Optional[str] = None,
        medical_conditions: Optional[list] = None,
        caregiver_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new user (patient/elderly)
        
        Returns:
            Dictionary with user profile (no password)
        """
        collection = await self._get_collection()
        
        # Check if email already exists
        existing = await collection.find_one({"email": email.lower().strip()})
        if existing:
            raise ValueError("Email already registered")
        
        # Generate user ID
        user_id = self.generate_user_id(full_name, age or 0)
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Prepare user document
        user_doc = {
            "user_id": user_id,
            "full_name": full_name.strip(),
            "email": email.lower().strip(),
            "phone_number": phone_number.strip() if phone_number else "",
            "age": age,
            "gender": gender or "",
            "address": address.strip() if address else "",
            "emergency_contact_name": emergency_contact_name.strip() if emergency_contact_name else "",
            "emergency_contact_number": emergency_contact_number.strip() if emergency_contact_number else "",
            "password": hashed_password,
            "profile_photo": profile_photo or "",
            "profile_photo_binary": None,
            "profile_photo_content_type": "",
            "medical_conditions": medical_conditions or [],
            "allergies": [],
            "special_treatments": [],
            "medicines": [],
            "medical_history": "",
            "caregiver_id": caregiver_id or "",
            "role": "user",
            "account_status": "active",
            "email_verified": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into database
        result = await collection.insert_one(user_doc)
        
        # Remove password from response
        user_doc.pop('password')
        user_doc['_id'] = str(result.inserted_id)
        
        logger.info(f"New user registered: {user_id}")
        
        return user_doc

    
    async def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user login with JWT tokens
        
        Args:
            email: User's email
            password: User's password
        
        Returns:
            Dictionary with user profile, JWT tokens, and authentication status
        """
        collection = await self._get_collection()
        
        # Find user by email
        user = await collection.find_one({"email": email.lower().strip()})
        
        if not user:
            raise ValueError("Invalid email or password")
        
        # Check if account is active
        if user.get('account_status') != 'active':
            raise ValueError("Account is not active. Please contact support.")
        
        # Verify password
        if not verify_password(password, user['password']):
            raise ValueError("Invalid email or password")
        
        # Update last login time
        await collection.update_one(
            {"_id": user['_id']},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Generate JWT tokens
        access_token = create_access_token(
            data={"user_id": user['user_id'], "role": "user"}
        )
        refresh_token = create_refresh_token(
            data={"user_id": user['user_id'], "role": "user"}
        )
        
        # Prepare response
        user.pop('password')
        user['_id'] = str(user['_id'])
        
        logger.info(f"User logged in: {user['user_id']}")
        
        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user
        }

    async def google_login(self, google_token: str) -> Dict[str, Any]:
        """
        Authenticate user via Google Sign-In.
        Verifies the Google ID token, creates user if new, and returns JWT tokens.
        """
        # Verify Google ID token
        try:
            idinfo = google_id_token.verify_oauth2_token(
                google_token, google_requests.Request(), GOOGLE_CLIENT_ID
            )
        except Exception as e:
            logger.error(f"Google token verification failed: {e}")
            raise ValueError("Invalid Google token")

        email = idinfo.get("email")
        if not email:
            raise ValueError("Google account has no email")

        name = idinfo.get("name", email.split("@")[0])
        picture = idinfo.get("picture", "")

        collection = await self._get_collection()

        # Check if user already exists
        user = await collection.find_one({"email": email.lower().strip()})

        if user:
            # Existing user - update last login
            if user.get("account_status") != "active":
                raise ValueError("Account is not active. Please contact support.")

            await collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            user.pop("password", None)
            user["_id"] = str(user["_id"])
        else:
            # New user - create account
            user_id = self.generate_user_id(name, 0)
            user_doc = {
                "user_id": user_id,
                "full_name": name,
                "email": email.lower().strip(),
                "phone_number": "",
                "age": None,
                "gender": "",
                "address": "",
                "emergency_contact_name": "",
                "emergency_contact_number": "",
                "password": "",
                "profile_photo": picture,
                "medical_conditions": [],
                "caregiver_id": "",
                "role": "user",
                "auth_provider": "google",
                "account_status": "active",
                "email_verified": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            result = await collection.insert_one(user_doc)
            user_doc.pop("password")
            user_doc["_id"] = str(result.inserted_id)
            user = user_doc
            logger.info(f"New Google user registered: {user_id}")

        # Generate JWT tokens
        uid = user.get("user_id", str(user["_id"]))
        access_token = create_access_token(data={"user_id": uid, "role": "user"})
        refresh_token = create_refresh_token(data={"user_id": uid, "role": "user"})

        logger.info(f"Google login successful: {email}")

        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user,
        }

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by user_id"""
        collection = await self._get_collection()
        user = await collection.find_one({"user_id": user_id})
        
        if user:
            user.pop('password', None)
            user.pop('profile_photo_binary', None)
            user['_id'] = str(user['_id'])
            # Add profile photo URL flag
            user['has_profile_photo'] = bool(user.get('profile_photo_content_type'))
            logger.info(f"Profile retrieved for user: {user_id}")
        else:
            logger.warning(f"User not found: {user_id}")
        
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user profile by email"""
        collection = await self._get_collection()
        user = await collection.find_one({"email": email.lower().strip()})
        
        if user:
            user.pop('password', None)
            user.pop('profile_photo_binary', None)
            user['_id'] = str(user['_id'])
        
        return user

    async def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile
        
        Args:
            user_id: User's unique ID
            update_data: Dictionary with fields to update
        
        Returns:
            Updated user profile
        """
        collection = await self._get_collection()
        
        # Fields that cannot be updated directly
        protected_fields = ['user_id', 'password', 'email', 'created_at', 'role']
        
        # Remove protected fields from update
        for field in protected_fields:
            update_data.pop(field, None)
        
        # Add updated timestamp
        update_data['updated_at'] = datetime.utcnow()
        
        # Perform update
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {"$set": update_data},
            return_document=True
        )
        
        if not result:
            logger.error(f"Cannot update: User not found {user_id}")
            raise ValueError(f"User {user_id} not found")
        
        result.pop('password', None)
        result.pop('profile_photo_binary', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"Profile updated for user: {user_id}")
        return result
    
    async def change_password(
        self, 
        user_id: str, 
        old_password: str, 
        new_password: str
    ) -> bool:
        """
        Change user password
        
        Args:
            user_id: User's unique ID
            old_password: Current password
            new_password: New password
        
        Returns:
            True if password changed successfully
        """
        collection = await self._get_collection()
        
        user = await collection.find_one({"user_id": user_id})
        
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Verify old password
        if not verify_password(old_password, user['password']):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        new_hashed_password = hash_password(new_password)
        
        # Update password
        await collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "password": new_hashed_password,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Password changed for user: {user_id}")
        
        return True

    async def delete_user(self, user_id: str) -> bool:
        """
        Delete user account (soft delete - marks as inactive)
        
        Args:
            user_id: User's unique ID
        
        Returns:
            True if deletion successful
        """
        collection = await self._get_collection()
        
        # Soft delete - update account status
        result = await collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "account_status": "deleted",
                    "deleted_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise ValueError(f"User not found: {user_id}")
        
        logger.info(f"User account deleted: {user_id}")
        return True
    
    async def link_to_caregiver(self, user_id: str, caregiver_id: str) -> Dict[str, Any]:
        """Link user to a caregiver"""
        collection = await self._get_collection()
        
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {
                    "caregiver_id": caregiver_id,
                    "updated_at": datetime.utcnow()
                }
            },
            return_document=True
        )
        
        if not result:
            raise ValueError(f"User not found: {user_id}")
        
        result.pop('password', None)
        result.pop('profile_photo_binary', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"User {user_id} linked to caregiver {caregiver_id}")
        
        return result

    async def upload_profile_photo(self, user_id: str, photo_base64: str, content_type: str = "image/jpeg") -> Dict[str, Any]:
        """
        Upload profile photo as binary data in MongoDB
        
        Args:
            user_id: User's unique ID
            photo_base64: Base64 encoded image data
            content_type: MIME type of the image
        
        Returns:
            Success response
        """
        collection = await self._get_collection()
        
        # Decode base64 to bytes
        try:
            # Remove data URL prefix if present
            if ',' in photo_base64:
                photo_base64 = photo_base64.split(',', 1)[1]
            photo_bytes = base64.b64decode(photo_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")
        
        # Check size limit (2MB)
        if len(photo_bytes) > 2 * 1024 * 1024:
            raise ValueError("Image size exceeds 2MB limit")
        
        # Store as Binary in MongoDB
        binary_data = Binary(photo_bytes)
        
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {
                    "profile_photo_binary": binary_data,
                    "profile_photo_content_type": content_type,
                    "updated_at": datetime.utcnow()
                }
            },
            return_document=True
        )
        
        if not result:
            raise ValueError(f"User not found: {user_id}")
        
        logger.info(f"Profile photo uploaded for user: {user_id}")
        
        return {"success": True, "message": "Profile photo uploaded successfully"}

    async def get_profile_photo(self, user_id: str) -> Optional[tuple]:
        """
        Get profile photo binary data
        
        Returns:
            Tuple of (photo_bytes, content_type) or None
        """
        collection = await self._get_collection()
        
        user = await collection.find_one(
            {"user_id": user_id},
            {"profile_photo_binary": 1, "profile_photo_content_type": 1}
        )
        
        if not user or not user.get("profile_photo_binary"):
            return None
        
        return (bytes(user["profile_photo_binary"]), user.get("profile_photo_content_type", "image/jpeg"))

    async def update_medical_records(self, user_id: str, records: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update medical records for a user
        
        Args:
            user_id: User's unique ID
            records: Dictionary with allergies, special_treatments, medicines, medical_history
        """
        collection = await self._get_collection()
        
        update_fields = {}
        if "allergies" in records:
            update_fields["allergies"] = records["allergies"]
        if "special_treatments" in records:
            update_fields["special_treatments"] = records["special_treatments"]
        if "medicines" in records:
            update_fields["medicines"] = records["medicines"]
        if "medical_history" in records:
            update_fields["medical_history"] = records["medical_history"]
        if "medical_conditions" in records:
            update_fields["medical_conditions"] = records["medical_conditions"]
        
        update_fields["updated_at"] = datetime.utcnow()
        
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {"$set": update_fields},
            return_document=True
        )
        
        if not result:
            raise ValueError(f"User not found: {user_id}")
        
        result.pop('password', None)
        result.pop('profile_photo_binary', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"Medical records updated for user: {user_id}")
        return result

    async def get_medical_records(self, user_id: str) -> Dict[str, Any]:
        """Get medical records for a user"""
        collection = await self._get_collection()
        
        user = await collection.find_one(
            {"user_id": user_id},
            {
                "allergies": 1, "special_treatments": 1,
                "medicines": 1, "medical_history": 1,
                "medical_conditions": 1
            }
        )
        
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        return {
            "allergies": user.get("allergies", []),
            "special_treatments": user.get("special_treatments", []),
            "medicines": user.get("medicines", []),
            "medical_history": user.get("medical_history", ""),
            "medical_conditions": user.get("medical_conditions", [])
        }

    async def calculate_profile_completion(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate profile completion percentage
        
        Tracked fields: full_name, phone_number, age, gender, address,
        emergency_contact_name, emergency_contact_number, profile_photo_binary,
        allergies, medicines, medical_history, caregiver_id
        """
        collection = await self._get_collection()
        user = await collection.find_one({"user_id": user_id})
        
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        fields_to_check = {
            "full_name": bool(user.get("full_name")),
            "phone_number": bool(user.get("phone_number")),
            "age": user.get("age") is not None,
            "gender": bool(user.get("gender")),
            "address": bool(user.get("address")),
            "emergency_contact_name": bool(user.get("emergency_contact_name")),
            "emergency_contact_number": bool(user.get("emergency_contact_number")),
            "profile_photo": bool(user.get("profile_photo_binary")),
            "allergies": bool(user.get("allergies")),
            "medicines": bool(user.get("medicines")),
            "medical_history": bool(user.get("medical_history")),
            "caregiver_id": bool(user.get("caregiver_id")),
        }
        
        completed = sum(1 for v in fields_to_check.values() if v)
        total = len(fields_to_check)
        percentage = round((completed / total) * 100)
        
        # Find missing fields
        missing = [k for k, v in fields_to_check.items() if not v]
        
        return {
            "percentage": percentage,
            "completed_fields": completed,
            "total_fields": total,
            "missing_fields": missing
        }

    async def get_user_sessions(self, user_id: str) -> list:
        logger.info(f"Retrieving sessions for user: {user_id}")
        return []

    async def create_user_session(self, user_id: str, session_type: str = "conversational") -> Dict[str, Any]:
        import uuid
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        session_data = {
            "session_id": session_id,
            "type": session_type,
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat()
        }
        if self.db is not None:
            collection = self.db["chat_sessions"]
            await collection.insert_one(session_data)
        logger.info(f"Session created for user {user_id}: {session_id}")
        return session_data

    async def validate_user(self, user_id: str) -> bool:
        user = await self.get_user_profile(user_id)
        return user is not None

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        logger.info(f"Retrieving stats for user: {user_id}")
        return {
            "user_id": user_id,
            "total_sessions": 0,
            "total_messages": 0,
            "last_activity": None,
            "average_session_duration": 0
        }

    async def generate_password_reset_code(self, email: str) -> str:
        """
        Generate and store a 6-digit password reset code
        
        Args:
            email: User's email address
        
        Returns:
            The generated reset code
        
        Raises:
            ValueError: If user not found
        """
        collection = await self._get_collection()
        
        # Check if user exists
        user = await collection.find_one({"email": email.lower().strip()})
        if not user:
            raise ValueError("No account found with this email address")
        
        # Generate 6-digit code
        reset_code = str(random.randint(100000, 999999))
        
        # Store reset code with expiry (15 minutes)
        expiry_time = datetime.utcnow() + timedelta(minutes=15)
        
        await collection.update_one(
            {"email": email.lower().strip()},
            {
                "$set": {
                    "password_reset_code": reset_code,
                    "password_reset_expiry": expiry_time,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Password reset code generated for {email}")
        
        return reset_code

    async def verify_reset_code(self, email: str, code: str) -> bool:
        """
        Verify if reset code is valid and not expired
        
        Args:
            email: User's email
            code: Reset code to verify
        
        Returns:
            True if code is valid, False otherwise
        """
        collection = await self._get_collection()
        
        user = await collection.find_one({"email": email.lower().strip()})
        
        if not user:
            return False
        
        # Check if code exists and matches
        if user.get('password_reset_code') != code:
            return False
        
        # Check if code has expired
        expiry = user.get('password_reset_expiry')
        if not expiry or datetime.utcnow() > expiry:
            return False
        
        return True

    async def reset_password_with_code(
        self,
        email: str,
        reset_code: str,
        new_password: str
    ) -> bool:
        """
        Reset password using reset code
        
        Args:
            email: User's email
            reset_code: Reset code
            new_password: New password
        
        Returns:
            True if password reset successfully
        
        Raises:
            ValueError: If code is invalid or expired
        """
        collection = await self._get_collection()
        
        # Verify reset code
        if not await self.verify_reset_code(email, reset_code):
            raise ValueError("Invalid or expired reset code")
        
        # Hash new password
        hashed_password = hash_password(new_password)
        
        # Update password and clear reset code
        result = await collection.update_one(
            {"email": email.lower().strip()},
            {
                "$set": {
                    "password": hashed_password,
                    "updated_at": datetime.utcnow()
                },
                "$unset": {
                    "password_reset_code": "",
                    "password_reset_expiry": ""
                }
            }
        )
        
        if result.modified_count == 0:
            raise ValueError("Failed to reset password")
        
        logger.info(f"Password reset successfully for {email}")
        
        return True



_user_service = None


def get_user_service() -> UserService:
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
