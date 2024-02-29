from bson import ObjectId
from app import config
from app.celery import celeryapp
from app.models.experiment import ExperimentModel, ExperimentCollection

from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from celery.result import AsyncResult
import motor.motor_asyncio

router = APIRouter()

client = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URL)
db = client.get_database('mnist-observability')
experiment_collection = db.get_collection('experiments')

@router.get(
    '/',
    response_description="List all experiments",
    response_model=ExperimentCollection,
    response_model_by_alias=False
)
async def get_all_experiments():
    return ExperimentCollection(experiments=await experiment_collection.find().to_list())

@router.get(
    '/{id}',
    response_description="Get a single experiment",
    response_model=ExperimentModel,
    response_model_by_alias=False
)
async def get_experiment_with_id(id: str):
    experiment = await experiment_collection.find_one({
        "_id": ObjectId(id)
    })

    if experiment is not None:
        return experiment
    
    raise HTTPException(404, "Experiment not found")

@router.post(
    '/',
    response_description="Add new experiment",
    response_model=ExperimentModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False
)
def create_new_experiment():
    pass

@router.post('/{id}/clone')
def clone_experiment(id: str):
    pass

@router.delete('/{id}')
def delete_experiment(id: str):
    pass

@router.websocket('/logs/{task_id}')
async def stream_logs(task_id: str, websocket: WebSocket):
    await websocket.accept()
