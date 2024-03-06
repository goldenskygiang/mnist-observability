import os
from typing import List
import dotenv

dotenv.load_dotenv()

MONGODB_URL = os.environ.get('MONGODB_URL')
DB_NAME = os.environ.get('DB_NAME')
EXPERIMENT_COLLECTION_NAME = 'experiments'

REDIS_URL = os.environ.get('REDIS_URL')

MNIST_DATASET_DIR = os.environ.get('MNIST_DATASET_DIR')
LOG_DIR = os.environ.get('LOG_DIR')

GRID_SEARCH_LIMIT = 16

# Celery configuration
class CeleryConfig:
    result_backend = MONGODB_URL
    broker_url = REDIS_URL
    task_acks_late = True

def init_dirs(dirs: List[str]):
    for d in dirs:
        if not(os.path.exists(d)):
            os.mkdir(d)

init_dirs([MNIST_DATASET_DIR, LOG_DIR])