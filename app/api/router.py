from fastapi import APIRouter

from app.api.endpoints import experiments

api_router = APIRouter()

api_router.include_router(experiments.router, prefix='/experiments')