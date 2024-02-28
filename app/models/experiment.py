from app.models.hyperparam import Hyperparam
from app.models.metrics import Metrics
from datetime import datetime
from typing import Annotated, Optional
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import List, Optional

PyObjectId = Annotated[str, BeforeValidator(str)]

class Experiment(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    name: str
    seed: Optional[int]
    celery_task_id: Optional[str]
    use_gpu: bool
    status: str
    deleted: bool = False
    log_dir: str
    checkpoint_dir: str
    hyperparam: Hyperparam
    last_result: Metrics
    result_per_epoch: List[Metrics]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __setattr__(self, name, value):
        super().__setattr__('updated_at', datetime.utcnow())
        super().__setattr__(name, value)

class ExperimentCollection(BaseModel):
    experiments: List[Experiment]