from fastapi import APIRouter
import time

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "LaQris API",
    }
