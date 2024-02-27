import os
import dotenv

dotenv.load_dotenv()

MONGODB_URL = os.environ.get('MONGODB_URL')

CELERY_BACKEND = MONGODB_URL
CELERY_BROKER = os.environ.get('REDIS_URL')