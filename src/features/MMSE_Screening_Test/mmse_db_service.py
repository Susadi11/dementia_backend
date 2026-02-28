# src/features/mmse_screening/mmse_db_service.py

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME")

if not MONGO_URL:
    raise RuntimeError("MONGODB_URI not set in environment variables")

_client = AsyncIOMotorClient(MONGO_URL)
_db = _client[DATABASE_NAME]


class MMSEDatabaseService:
    def __init__(self):
        self.users = _db["users"]
        self.assessments = _db["assessments"]

    #User 
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self.users.find_one({"user_id": user_id})

    #Assessment lifecycle
    async def create_assessment(self, user_id: str) -> str:
        doc = {
            "user_id": user_id,
            "assessment_type": "MMSE",
            "assessment_date": datetime.utcnow(),
            "questions": [],
            "total_score": 0,
            "ml_summary": {},
            "status": "in_progress",
        }
        result = await self.assessments.insert_one(doc)
        return str(result.inserted_id)

    async def add_question(
        self,
        assessment_id: str,
        user_id: str,
        question_doc: Dict[str, Any],
    ):
        await self.assessments.update_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id},
            {"$push": {"questions": question_doc}},
        )

    async def get_assessment(self, assessment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        if not ObjectId.is_valid(assessment_id):
            return None

        return await self.assessments.find_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id}
        )

    async def finalize_assessment(
        self,
        assessment_id: str,
        total_score: float,
        avg_ml_prob: float,
        ml_label: str,
    ):
        await self.assessments.update_one(
            {"_id": ObjectId(assessment_id)},
            {
                "$set": {
                    "total_score": total_score,
                    "ml_summary": {
                        "avg_probability": avg_ml_prob,
                        "ml_risk_label": ml_label,
                    },
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                }
            },
        )


# singleton instance for use across the service
db_service = MMSEDatabaseService()