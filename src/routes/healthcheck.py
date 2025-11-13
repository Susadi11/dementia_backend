from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "dementia_backend",
        "message": "API is running successfully"
    }


@router.get("/status")
async def system_status() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "modules": {
            "conversational_ai": "active",
            "database": "connected",
            "preprocessing": "ready"
        }
    }
