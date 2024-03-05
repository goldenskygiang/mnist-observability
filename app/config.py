import os
from typing import List
import dotenv

dotenv.load_dotenv()

def init_dirs(dirs: List[str]):
    for d in dirs:
        if not(os.path.exists(d)):
            os.mkdir(d)

MONGODB_URL = os.environ.get('MONGODB_URL')
REDIS_URL = os.environ.get('REDIS_URL')

MNIST_DATASET_DIR = os.environ.get('MNIST_DATASET_DIR')
LOG_DIR = os.environ.get('LOG_DIR')

init_dirs([MNIST_DATASET_DIR, LOG_DIR])

GRID_SEARCH_LIMIT = 16

# Celery configuration
result_backend = MONGODB_URL
broker_url = REDIS_URL
task_acks_late = True