import uvicorn
from fastapi import FastAPI

from app.api.router import api_router
from app.celery import celeryapp

def get_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    return app

app = get_app()

if __name__ == 'main':
    uvicorn.run(app, host='0.0.0.0', port=8080)