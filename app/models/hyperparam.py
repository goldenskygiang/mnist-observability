from app.models import AbstractBaseModel

class Hyperparam(AbstractBaseModel):
    epoches: int
    learning_rate: float
    dropout: float
    batch_size: int
    optimizer: str
    loss_func: str