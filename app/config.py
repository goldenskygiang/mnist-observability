import os
import dotenv

dotenv.load_dotenv()

MONGODB_URL = os.environ.get('MONGODB_URL')
MNIST_DATASET_DIR = os.environ.get('MNIST_DATASET_DIR')
REDIS_URL = os.environ.get('REDIS_URL')
LOG_DIR = os.environ.get('LOG_DIR')

GRID_SEARCH_LIMIT = 16

# Celery configuration
result_backend = MONGODB_URL
broker_url = REDIS_URL
task_acks_late = True