from datetime import datetime
import logging
import os
import time
from app import config
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
    exp_obj_id = ObjectId(exp_id)

    log_path = os.path.abspath(os.path.join(config.LOG_DIR, f'{exp_id}.txt'))
    with open(log_path, 'w') as f:
        pass

    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    ))
    logger.addHandler(fh)

    start_time = time.time()

    try:
        exp = loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            {
                "$set":
                {
                    "celery_task_id": task_id,
                    "status": ExperimentStatus.RUNNING,
                    "log_dir": log_path
                }
            },
            return_document=ReturnDocument.AFTER
        ))

        exp = ExperimentModel.model_validate(exp)
        args = exp.hyperparam

        train_results = {}
        test_results = {}

        for lr in args.learning_rate:
            for batch_sz in args.batch_size:
                key = f"lr={lr},batch_size={batch_sz}"

                logger.info(f'Configuration: {key}')

                model = init_linear_model(args.output_activation_func)
                model, train_metrics_per_epoch = train_model(model, exp, lr, batch_sz)
                model, test_metrics = test_model(model, exp, lr, batch_sz)

                train_results[key] = [
                    {
                        k: v for k, v in ep.model_dump(by_alias=True).items()
                    } for ep in train_metrics_per_epoch
                ]

                test_results[key] = {
                    k: v for k, v in test_metrics.model_dump(by_alias=True).items()
                }

        upd = {
            "train_results": train_results,
            "test_result": test_results,
            "status": ExperimentStatus.SUCCESS
        }

        loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            { "$set": upd },
            return_document=ReturnDocument.AFTER
        ))

    except Exception as e:
        logger.exception(e)
        
        loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            {
                "$set":
                {
                    "celery_task_id": task_id,
                    "status": ExperimentStatus.ERROR,
                    "log_dir": log_path
                }
            },
            return_document=ReturnDocument.AFTER
        ))
        
    finally:
        loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            {
                "$set":
                {
                    "updated_at": datetime.utcnow(),
                    "elapsed_time": time.time() - start_time
                }
            },
            return_document=ReturnDocument.AFTER
        ))