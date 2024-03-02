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