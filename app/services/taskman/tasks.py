from app.mnist.linear import init_linear_model, test_model, train_model
from app.models.enums import ExperimentStatus
from app.models import experiment_collection
from app.celery import celeryapp

from bson import ObjectId
from pymongo import ReturnDocument
import asyncio

from app.models.experiment import ExperimentModel

@celeryapp.task(bind=True)
def start_experiment(self, exp_id: str):
    loop = asyncio.get_event_loop()

    task_id = self.request.id
    exp_id = ObjectId(exp_id)

    exp = loop.run_until_complete(experiment_collection.find_one_and_update(
        { "_id": exp_id },
        { "$set": { "celery_task_id": task_id } },
        return_document=ReturnDocument.AFTER
    ))

    exp = ExperimentModel.model_validate(exp)
    args = exp.hyperparam

    model = init_linear_model(args.output_activation_func)
    model, train_metrics_per_epoch = train_model(model, exp)
    model, test_metrics = test_model(model, exp)

    test_metrics = {
        k: v for k, v in test_metrics.model_dump(by_alias=True).items()
    }

    train_metrics_per_epoch = [
        {
            k: v for k, v in ep.model_dump(by_alias=True).items()
        } for ep in train_metrics_per_epoch
    ]

    upd = {
        "train_results": train_metrics_per_epoch,
        "test_result": test_metrics,
        "status": ExperimentStatus.SUCCESS
    }

    loop.run_until_complete(experiment_collection.find_one_and_update(
        { "_id": exp_id },
        { "$set": upd },
        return_document=ReturnDocument.AFTER
    ))

@celeryapp.task
def stop_experiment():
    pass

def get_job_status():
    pass