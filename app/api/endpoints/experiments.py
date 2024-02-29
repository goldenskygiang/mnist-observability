from app import config
from app.celery import celeryapp
from app.models.experiment import ExperimentCollection

from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult
import motor.motor_asyncio

import os
import uuid

router = APIRouter()

client = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URL)
db = client.get_database('mnist-observability')
experiment_collection = db.get_collection('experiments')

@router.get('/')
async def get_all_experiments():
    return ExperimentCollection(experiments=await experiment_collection.find().to_list(1000))

@router.get('/{id}')
def get_experiment_with_id(id: str):
    pass

@router.post('/')
def create_new_experiment():
    pass

@router.post('/{id}/clone')
def clone_experiment(id: str):
    pass

@router.delete('/{id}')
def delete_experiment(id: str):
    pass