import app.config as config
import motor.motor_asyncio
from pymongo import MongoClient

__async_client__ = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URL)
async_db = __async_client__.get_database(config.DB_NAME)

__client__ = MongoClient(config.MONGODB_URL)
db = __client__.get_database(config.DB_NAME)
