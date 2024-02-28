from app.models.enums import *
from typing import List

class Hyperparam:
    epochs: int
    learning_rate: List[float]
    dropout: List[float]
    batch_size: List[int]
    optimizer: Optimizer
    output_activation_func: ActivationFunction
    loss_func: LossFunction
