from app.models.enums import ExperimentStatus
from app.models.hyperparam import Hyperparam
from app.models.metrics import Metrics

from datetime import datetime
from typing import Annotated, Optional, List
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

PyObjectId = Annotated[str, BeforeValidator(str)]

class ExperimentDto(BaseModel):
    name: str
    seed: Optional[int] = None
    use_gpu: bool = False
    hyperparam: Hyperparam

class ExperimentModel(ExperimentDto):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    celery_task_id: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    log_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    test_result: Optional[Metrics] = None
    train_results: Optional[List[Metrics]] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    def __setattr__(self, name, value):
        super().__setattr__('updated_at', datetime.utcnow())
        super().__setattr__(name, value)

class ExperimentCollection(BaseModel):
    experiments: List[ExperimentModel]

"""
        json_schema_extra={
            "example": {
                "name": "MNIST 1",
                "use_gpu": False,
                "status": "created",
                "hyperparam": {
                    "epoches": 5,
                    "learning_rate": [0.01],
                    "dropout": [],
                    "batch_size": [128],
                    "optimizer": "SGD",
                    "output_activation_func": "softmax",
                    "loss_func": "CrossEntropy"
                }
            }
        }
"""