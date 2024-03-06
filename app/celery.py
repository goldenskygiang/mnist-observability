from celery import Celery
import app.services
import app.config as config

celeryapp = Celery('app.services')
celeryapp.config_from_object(config.CeleryConfig)
celeryapp.autodiscover_tasks(['app.services.taskman'])