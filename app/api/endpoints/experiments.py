import asyncio
from bson import ObjectId
from pymongo import ReturnDocument
import pymongo
from app import config
from app.celery import celeryapp
from app.models.enums import ExperimentStatus
from app.services.taskman import tasks as taskman
from app.models import experiment_collection
from app.models.experiment import ExperimentDto, ExperimentModel, ExperimentCollection

from fastapi import APIRouter, Body, Depends, HTTPException, Response, WebSocket, WebSocketException, status
from celery.result import AsyncResult

router = APIRouter()

@router.get(
    '/',
    response_description="List all experiments",
    response_model=ExperimentCollection,
    response_model_by_alias=False
)
async def get_all_experiments():
    return ExperimentCollection(
        experiments=await experiment_collection.find().sort(
            "updated_at", pymongo.DESCENDING).to_list(1000))

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
    runs = len(exp.hyperparam.learning_rates) * len(exp.hyperparam.batch_sizes)

    if (runs < 1) or (runs > config.GRID_SEARCH_LIMIT):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Total combinations of grid search must be <= {config.GRID_SEARCH_LIMIT}!")

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

@router.websocket('/ws/logs')
async def stream_logs(websocket: WebSocket):
    await websocket.accept()

    id = await websocket.receive_text()

    experiment = await experiment_collection.find_one(
        { "_id": ObjectId(id) }
    )

    if experiment is None:
        raise WebSocketException(
            code=status.WS_1013_TRY_AGAIN_LATER,
            reason=f"Experiment id {id} not found")
    
    experiment = ExperimentModel.model_validate(experiment)

    if experiment.log_dir is not None:
        with open(experiment.log_dir, 'r') as f:
            await websocket.send_text(f.read())
            while True:
                content = f.read()
                if content:
                    await websocket.send_text(content)
                    experiment = ExperimentModel.model_validate(
                        await experiment_collection.find_one(
                            { "_id": ObjectId(id) }
                        )
                    )
                else:
                    await asyncio.sleep(5)

                if experiment.status == ExperimentStatus.SUCCESS or \
                experiment.status == ExperimentStatus.ERROR:
                    break

    await websocket.close()