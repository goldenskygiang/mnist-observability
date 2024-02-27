from app.celery import celeryapp

from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult

import os
import uuid

router = APIRouter()

@router.get('/')
def get_all_experiments():
    pass

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