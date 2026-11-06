import os
import platform
import time
from fastapi import APIRouter
from app.config import settings

router = APIRouter()
start_time = time.time()

@router.get("/health", summary="Health Check")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "uptime_seconds": round(time.time() - start_time, 2),
        "system": {
            "hostname": platform.node(),
            "os": platform.system(),
            "python": platform.python_version()
        }
    }
