from app.models import AbstractBaseModel
from app.models.hyperparam import Hyperparam
from app.models.metrics import Metrics

from typing import List, Optional

class Experiment(AbstractBaseModel):
    name: str
    seed: Optional[int]
    celery_task_id: Optional[str]
    use_gpu: bool
    status: str
    train_ratio: float = 0.7
    deleted: bool = False
    log_dir: str
    checkpoint_dir: str
    hyperparam: Hyperparam
    last_result: Metrics
    result_per_epoch: List[Metrics]

class ExperimentCollection(AbstractBaseModel):
    experiments: List[Experiment]