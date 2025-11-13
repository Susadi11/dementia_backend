from typing import Dict, Any, Optional
import logging
from datetime import datetime
from .db_service import get_db_service

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self):
        self.db_service = get_db_service()
        logger.info("User service initialized")

    async def register_user(self, email: str, name: str, age: Optional[int] = None) -> Dict[str, Any]:
        user_data = {
            "email": email,
            "name": name,
            "age": age,
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        created_user = await self.db_service.create_user(user_data)
        logger.info(f"User registered: {created_user.get('id')}")
        return created_user

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = await self.db_service.get_user(user_id)
        if not user:
            logger.warning(f"User not found: {user_id}")
            return None
        logger.info(f"Profile retrieved for user: {user_id}")
        return user

    async def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        user = await self.db_service.get_user(user_id)
        if not user:
            logger.error(f"Cannot update: User not found {user_id}")
            raise ValueError(f"User {user_id} not found")

        updated_user = await self.db_service.update_user(user_id, update_data)
        logger.info(f"Profile updated for user: {user_id}")
        return updated_user

    async def delete_user(self, user_id: str) -> bool:
        success = await self.db_service.delete_user(user_id)
        if success:
            logger.info(f"User deleted: {user_id}")
        return success

    async def get_user_sessions(self, user_id: str) -> list:
        logger.info(f"Retrieving sessions for user: {user_id}")
        return []

    async def create_user_session(self, user_id: str, session_type: str = "conversational") -> Dict[str, Any]:
        session_data = {
            "type": session_type,
            "user_id": user_id,
            "started_at": datetime.now().isoformat()
        }
        session = await self.db_service.create_session(user_id, session_data)
        logger.info(f"Session created for user {user_id}: {session.get('session_id')}")
        return session

    async def validate_user(self, user_id: str) -> bool:
        user = await self.db_service.get_user(user_id)
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


_user_service = None


def get_user_service() -> UserService:
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
