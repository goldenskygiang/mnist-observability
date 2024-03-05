from pydantic import BaseModel, ConfigDict
from app.models.enums import *
from typing import List

class Hyperparam(BaseModel):
    epochs: int
    learning_rates: List[float]
    batch_sizes: List[int]
    optimizer: Optimizer
    output_activation_func: ActivationFunction

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )
