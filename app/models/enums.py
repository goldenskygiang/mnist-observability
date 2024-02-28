from enum import Enum

class Optimizer(str, Enum):
    Adam = "Adam"
    SGD = "SGD"
    RMSprop = "RMSprop"

class ActivationFunction(str, Enum):
    softmax = "softmax"
    tanh = "tanh"

class LossFunction(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    CrossEntropy = "CrossEntropy"
