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

@celeryapp.task(bind=True, ignore_result=True)
def start_experiment(self, exp_id: str):
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
        try:
            logger.info('Obtaining event loop')
            loop = asyncio.get_running_loop()
        except Exception as e:
            logger.exception(e)
            logger.info('Obtaining new event loop')
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

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

        run_cnt = 0
        total_runs = len(args.learning_rates) * len(args.batch_sizes)

        logger.info(f'Experiment {exp_id} started')

        for lr in args.learning_rates:
            for batch_sz in args.batch_sizes:
                run_cnt += 1
                key = f"[{run_cnt:02d}/{total_runs:02d}] Learning Rate = {lr} | Batch size = {batch_sz}"
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
            "test_results": test_results,
            "status": ExperimentStatus.SUCCESS
        }

        logger.info(f'Experiment {exp_id} completed successfully')

        loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            { "$set": upd },
            return_document=ReturnDocument.AFTER
        ))

    except Exception as e:
        logger.exception(e)
        logger.info(f'Experiment {exp_id} failed')
        
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
        elapsed_time = time.time() - start_time
        loop.run_until_complete(experiment_collection.find_one_and_update(
            { "_id": exp_obj_id },
            {
                "$set":
                {
                    "updated_at": datetime.utcnow(),
                    "elapsed_time": elapsed_time
                }
            },
            return_document=ReturnDocument.AFTER
        ))

        logger.info(f'Elapsed time: {elapsed_time} s')