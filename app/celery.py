from celery import Celery
from app.mnist.dataset import init_dataset
import app.services
import app.config as config

init_dataset(config.MNIST_DATASET_DIR)

celeryapp = Celery('app.services')
celeryapp.config_from_object(config)
celeryapp.autodiscover_tasks(['app.services.taskman'])