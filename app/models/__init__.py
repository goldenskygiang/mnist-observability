from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId

class AbstractBaseModel(BaseModel):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def __setattr__(self, name, value):
        super().__setattr__('updated_at', datetime.utcnow())
        super().__setattr__(name, value)
