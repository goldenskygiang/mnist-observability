import app.config as config
import motor.motor_asyncio

__client__ = motor.motor_asyncio.AsyncIOMotorClient(config.MONGODB_URL)
DB = __client__.get_database('mnist-observability')

experiment_collection = DB.get_collection('experiments')
