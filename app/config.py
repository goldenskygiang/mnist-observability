import os
import dotenv

dotenv.load_dotenv()

MONGODB_URL = os.environ.get('MONGODB_URL')
MNIST_DATASET_DIR = os.environ.get('MNIST_DATASET_DIR')

LOG_DIR = os.environ.get('LOG_DIR')

CELERY_BACKEND = MONGODB_URL
CELERY_BROKER = os.environ.get('REDIS_URL')