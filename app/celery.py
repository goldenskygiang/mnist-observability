from celery import Celery
import nest_asyncio
from app.mnist.dataset import init_dataset
import app.services
import app.config as config

init_dataset(config.MNIST_DATASET_DIR)

nest_asyncio.apply()

celeryapp = Celery('app.services')
celeryapp.config_from_object(config)
celeryapp.autodiscover_tasks(['app.services.taskman'])