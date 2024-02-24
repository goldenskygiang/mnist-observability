import os
import dotenv

dotenv.load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')

CELERY_BACKEND = f'db+{DATABASE_URL}'
CELERY_BROKER = os.environ.get('REDIS_URL')