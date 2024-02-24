from celery import Celery
#import app.services
import app.config as config

celeryapp = Celery('app.services', broker=config.CELERY_BROKER, backend=config.CELERY_BACKEND)
#celeryapp.autodiscover_tasks(['app.services.ocr'])