"""
Caregiver Service
Handles caregiver registration, authentication, and profile management
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
import re
from ..utils.auth import (
    hash_password, 
    verify_password, 
    create_tokens_for_caregiver
)

logger = logging.getLogger(__name__)


class CaregiverService:
    """Service for managing caregiver accounts"""
    
    def __init__(self, db):
        self.db = db
        self.collection = None
        logger.info("Caregiver service initialized")
    
    async def _get_collection(self):
        """Get or create caregivers collection"""
        if self.collection is None:
            self.collection = self.db.get_collection("caregivers")
        return self.collection
    
    def generate_caregiver_id(self, first_name: str, gender: str, mobile_number: str) -> str:
        """
        Generate unique caregiver ID in format: C<Gender-Initial>-<FIRSTNAME>-<Last4Digits>
        Example: CF-SUSADI-1567
        
        Args:
            first_name: Caregiver's first name
            gender: Gender (Male/Female/Other)
            mobile_number: Mobile phone number
        
        Returns:
            Formatted caregiver ID
        """
        # Get gender initial (F for Female, M for Male, O for Other)
        gender_initial = gender[0].upper() if gender else 'O'
        
        # Clean and uppercase first name
        clean_name = first_name.strip().upper().replace(" ", "")
        
        # Extract last 4 digits from mobile number
        # Remove any non-digit characters first
        digits_only = re.sub(r'\D', '', mobile_number)
        last_4_digits = digits_only[-4:] if len(digits_only) >= 4 else digits_only.zfill(4)
        
        # Format: C<Gender>-<NAME>-<DIGITS>
        caregiver_id = f"C{gender_initial}-{clean_name}-{last_4_digits}"
        
        return caregiver_id
    
    async def register_caregiver(self, caregiver_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new caregiver
        
        Args:
            caregiver_data: Dictionary containing caregiver registration information
                - first_name (required)
                - last_name (required)
                - nic_number (required)
                - mobile_number (required)
                - district (required)
                - gender (required)
                - email (required) - used as username
                - password (required)
                - profile_photo (optional)
                - emergency_contact_name (required)
                - emergency_contact_number (required)
                - declaration_accepted (required)
        
        Returns:
            Dictionary with caregiver profile (without password)
        """
        collection = await self._get_collection()
        
        # Validate required fields
        required_fields = [
            'first_name', 'last_name', 'nic_number', 'mobile_number',
            'district', 'gender', 'email', 'password',
            'emergency_contact_name', 'emergency_contact_number',
            'declaration_accepted'
        ]
        
        for field in required_fields:
            if field not in caregiver_data or not caregiver_data[field]:
                raise ValueError(f"Missing required field: {field}")
        
        # Check if declaration is accepted
        if not caregiver_data['declaration_accepted']:
            raise ValueError("Declaration must be accepted to register")
        
        # Check if email already exists
        existing_caregiver = await collection.find_one({"email": caregiver_data['email']})
        if existing_caregiver:
            raise ValueError("Email already registered")
        
        # Check if NIC already exists
        existing_nic = await collection.find_one({"nic_number": caregiver_data['nic_number']})
        if existing_nic:
            raise ValueError("NIC number already registered")
        
        # Generate unique caregiver ID
        caregiver_id = self.generate_caregiver_id(
            caregiver_data['first_name'],
            caregiver_data['gender'],
            caregiver_data['mobile_number']
        )
        
        # Check if generated ID already exists (unlikely but possible)
        counter = 1
        original_id = caregiver_id
        while await collection.find_one({"caregiver_id": caregiver_id}):
            caregiver_id = f"{original_id}-{counter}"
            counter += 1
        
        # Hash password
        hashed_password = hash_password(caregiver_data['password'])
        
        # Prepare caregiver document
        caregiver_doc = {
            "caregiver_id": caregiver_id,
            "first_name": caregiver_data['first_name'].strip(),
            "last_name": caregiver_data['last_name'].strip(),
            "full_name": f"{caregiver_data['first_name'].strip()} {caregiver_data['last_name'].strip()}",
            "nic_number": caregiver_data['nic_number'].strip(),
            "mobile_number": caregiver_data['mobile_number'].strip(),
            "district": caregiver_data['district'].strip(),
            "gender": caregiver_data['gender'],
            "email": caregiver_data['email'].lower().strip(),
            "password": hashed_password,
            "profile_photo": caregiver_data.get('profile_photo', ''),
            "emergency_contact_name": caregiver_data['emergency_contact_name'].strip(),
            "emergency_contact_number": caregiver_data['emergency_contact_number'].strip(),
            "declaration_accepted": True,
            "declaration_accepted_at": datetime.utcnow(),
            "role": "caregiver",
            "account_status": "active",
            "email_verified": False,
            "patient_ids": [],  # Will be linked to patients later
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_login": None
        }
        
        # Insert into database
        result = await collection.insert_one(caregiver_doc)
        
        if not result.inserted_id:
            raise Exception("Failed to create caregiver account")
        
        # Remove password from response
        caregiver_doc.pop('password', None)
        caregiver_doc['_id'] = str(result.inserted_id)
        
        logger.info(f"Caregiver registered successfully: {caregiver_id}")
        
        return caregiver_doc
    
    async def login_caregiver(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate caregiver login with JWT tokens
        
        Args:
            email: Caregiver's email (username)
            password: Caregiver's password
        
        Returns:
            Dictionary with caregiver profile, JWT tokens, and authentication status
        """
        collection = await self._get_collection()
        
        # Find caregiver by email
        caregiver = await collection.find_one({"email": email.lower().strip()})
        
        if not caregiver:
            raise ValueError("Invalid email or password")
        
        # Check if account is active
        if caregiver.get('account_status') != 'active':
            raise ValueError("Account is not active. Please contact support.")
        
        # Verify password
        if not verify_password(password, caregiver['password']):
            raise ValueError("Invalid email or password")
        
        # Update last login time
        await collection.update_one(
            {"_id": caregiver['_id']},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Remove password from response
        caregiver.pop('password', None)
        caregiver['_id'] = str(caregiver['_id'])
        
        # Generate JWT tokens
        tokens = create_tokens_for_caregiver(caregiver)
        
        logger.info(f"Caregiver logged in: {caregiver['caregiver_id']}")
        
        return {
            "success": True,
            "message": "Login successful",
            "caregiver": caregiver,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": tokens["token_type"],
            "expires_in": tokens["expires_in"]
        }
    
    async def get_caregiver_by_id(self, caregiver_id: str) -> Optional[Dict[str, Any]]:
        """Get caregiver profile by caregiver_id"""
        collection = await self._get_collection()
        caregiver = await collection.find_one({"caregiver_id": caregiver_id})
        
        if caregiver:
            caregiver.pop('password', None)
            caregiver['_id'] = str(caregiver['_id'])
        
        return caregiver
    
    async def get_caregiver_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get caregiver profile by email"""
        collection = await self._get_collection()
        caregiver = await collection.find_one({"email": email.lower().strip()})
        
        if caregiver:
            caregiver.pop('password', None)
            caregiver['_id'] = str(caregiver['_id'])
        
        return caregiver
    
    async def update_caregiver_profile(
        self, 
        caregiver_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update caregiver profile
        
        Args:
            caregiver_id: Caregiver's unique ID
            update_data: Dictionary with fields to update
        
        Returns:
            Updated caregiver profile
        """
        collection = await self._get_collection()
        
        # Fields that cannot be updated directly
        protected_fields = ['caregiver_id', 'password', 'email', 'nic_number', 
                          'created_at', 'declaration_accepted', 'declaration_accepted_at']
        
        # Remove protected fields from update
        for field in protected_fields:
            update_data.pop(field, None)
        
        # Add updated timestamp
        update_data['updated_at'] = datetime.utcnow()
        
        # Update full_name if first_name or last_name changed
        if 'first_name' in update_data or 'last_name' in update_data:
            caregiver = await collection.find_one({"caregiver_id": caregiver_id})
            if caregiver:
                first = update_data.get('first_name', caregiver.get('first_name', ''))
                last = update_data.get('last_name', caregiver.get('last_name', ''))
                update_data['full_name'] = f"{first} {last}".strip()
        
        # Perform update
        result = await collection.find_one_and_update(
            {"caregiver_id": caregiver_id},
            {"$set": update_data},
            return_document=True
        )
        
        if not result:
            raise ValueError(f"Caregiver not found: {caregiver_id}")
        
        result.pop('password', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"Caregiver profile updated: {caregiver_id}")
        
        return result
    
    async def change_password(
        self, 
        caregiver_id: str, 
        old_password: str, 
        new_password: str
    ) -> bool:
        """
        Change caregiver password
        
        Args:
            caregiver_id: Caregiver's unique ID
            old_password: Current password
            new_password: New password
        
        Returns:
            True if password changed successfully
        """
        collection = await self._get_collection()
        
        # Get caregiver
        caregiver = await collection.find_one({"caregiver_id": caregiver_id})
        if not caregiver:
            raise ValueError("Caregiver not found")
        
        # Verify old password
        if not verify_password(old_password, caregiver['password']):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        new_hashed = hash_password(new_password)
        
        # Update password
        await collection.update_one(
            {"caregiver_id": caregiver_id},
            {
                "$set": {
                    "password": new_hashed,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Password changed for caregiver: {caregiver_id}")
        
        return True
    
    async def link_patient(self, caregiver_id: str, patient_id: str) -> Dict[str, Any]:
        """Link a patient to a caregiver"""
        collection = await self._get_collection()
        
        result = await collection.find_one_and_update(
            {"caregiver_id": caregiver_id},
            {
                "$addToSet": {"patient_ids": patient_id},
                "$set": {"updated_at": datetime.utcnow()}
            },
            return_document=True
        )
        
        if not result:
            raise ValueError(f"Caregiver not found: {caregiver_id}")
        
        result.pop('password', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"Patient {patient_id} linked to caregiver {caregiver_id}")
        
        return result
    
    async def unlink_patient(self, caregiver_id: str, patient_id: str) -> Dict[str, Any]:
        """Unlink a patient from a caregiver"""
        collection = await self._get_collection()
        
        result = await collection.find_one_and_update(
            {"caregiver_id": caregiver_id},
            {
                "$pull": {"patient_ids": patient_id},
                "$set": {"updated_at": datetime.utcnow()}
            },
            return_document=True
        )
        
        if not result:
            raise ValueError(f"Caregiver not found: {caregiver_id}")
        
        result.pop('password', None)
        result['_id'] = str(result['_id'])
        
        logger.info(f"Patient {patient_id} unlinked from caregiver {caregiver_id}")
        
        return result
    
    async def delete_caregiver(self, caregiver_id: str) -> bool:
        """
        Delete caregiver account (soft delete - marks as inactive)
        
        Args:
            caregiver_id: Caregiver's unique ID
        
        Returns:
            True if deletion successful
        """
        collection = await self._get_collection()
        
        # Soft delete - update account status
        result = await collection.update_one(
            {"caregiver_id": caregiver_id},
            {
                "$set": {
                    "account_status": "deleted",
                    "deleted_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise ValueError(f"Caregiver not found: {caregiver_id}")
        
        logger.info(f"Caregiver account deleted: {caregiver_id}")
        
        return True


# Singleton instance
_caregiver_service = None


def get_caregiver_service(db=None):
    """Get or create caregiver service instance"""
    global _caregiver_service
    if _caregiver_service is None and db is not None:
        _caregiver_service = CaregiverService(db)
    return _caregiver_service
