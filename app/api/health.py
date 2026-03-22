from fastapi import APIRouter
from pydantic import BaseModel
from config.settings import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    env: str
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", env=settings.app_env)
