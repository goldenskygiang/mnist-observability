from pydantic import BaseModel, ConfigDict
from app.models.enums import *
from typing import List

class Hyperparam(BaseModel):
    epochs: int
    learning_rate: List[float]
    dropout: List[float]
    batch_size: List[int]
    optimizer: Optimizer
    output_activation_func: ActivationFunction
    loss_func: LossFunction

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )
