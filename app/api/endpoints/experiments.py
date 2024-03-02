from bson import ObjectId
from pymongo import ReturnDocument
from app import config
from app.celery import celeryapp
from app.services.taskman import tasks as taskman
from app.models import experiment_collection
from app.models.experiment import ExperimentDto, ExperimentModel, ExperimentCollection

from fastapi import APIRouter, Body, Depends, HTTPException, Response, WebSocket, status
from celery.result import AsyncResult
import motor.motor_asyncio

router = APIRouter()

@router.get(
    '/',
    response_description="List all experiments",
    response_model=ExperimentCollection,
    response_model_by_alias=False
)
async def get_all_experiments():
    return ExperimentCollection(experiments=await experiment_collection.find().to_list(1000))

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
async def create_new_experiment(exp: ExperimentDto = Body()):
    new_exp = await experiment_collection.insert_one(
        exp.model_dump(by_alias=True)
    )

    created_exp = await experiment_collection.find_one(
        { "_id": new_exp.inserted_id }
    )

    taskman.start_experiment.delay(str(new_exp.inserted_id))
    
    return created_exp

@router.delete(
    '/{id}',
    response_description="Delete an experiment"
)
async def delete_experiment(id: str):
    delete_result = await experiment_collection.delete_one(
        { "_id": ObjectId(id) }
    )

    if delete_result.deleted_count == 1:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment {id} not found")

@router.websocket('/logs/{task_id}')
async def stream_logs(task_id: str, websocket: WebSocket):
    await websocket.accept()
