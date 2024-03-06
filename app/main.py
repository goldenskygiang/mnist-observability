from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI

from app.api.router import api_router
from app.celery import celeryapp
import app.config as config
from app.mnist.dataset import init_dataset

def get_app() -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.include_router(api_router)
    return app

init_dataset()

app = get_app()

if __name__ == 'main':
    uvicorn.run(app, host='0.0.0.0', port=8080)