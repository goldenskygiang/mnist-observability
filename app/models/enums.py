from enum import Enum

class Optimizer(str, Enum):
    Adam = "Adam"
    SGD = "SGD"
    RMSprop = "RMSprop"

class ActivationFunction(str, Enum):
    softmax = "softmax"
    tanh = "tanh"

class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"