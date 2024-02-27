from datetime import datetime
from typing import Optional, List
from typing_extensions import Annotated

from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator

from bson import ObjectId
import motor.motor_asyncio

PyObjectId = Annotated[str, BeforeValidator(str)]

class AbstractBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = datetime.now()

    class Config:
        pass