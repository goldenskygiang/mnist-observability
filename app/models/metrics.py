from pydantic import BaseModel, ConfigDict

class Metrics(BaseModel):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    runtime: float

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    def __init__(self, loss, acc: float, pre: float, rec: float, f1: float, runtime: float):
        self.loss = loss
        self.accuracy = acc
        self.precision = pre
        self.recall = rec
        self.f1_score = f1
        self.runtime = runtime