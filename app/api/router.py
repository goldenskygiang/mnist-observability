from fastapi import APIRouter

from app.api import pages
from app.api.endpoints import experiments

api_router = APIRouter()

api_router.include_router(pages.router, prefix='')
api_router.include_router(experiments.router, prefix='/api/experiments')