from app.models.enums import ExperimentStatus
from app.models.hyperparam import Hyperparam
from app.models.metrics import Metrics

from datetime import datetime
from typing import Annotated, Dict, Optional, List
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
    test_results: Optional[Dict[str, Metrics]] = None
    train_results: Optional[Dict[str, List[Metrics]]] = None

    elapsed_time: float = 0

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

class ExperimentCollection(BaseModel):
    experiments: List[ExperimentModel]
