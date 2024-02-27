class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    runtime: float

    def __init__(self, truths, predictions):
        assert(truths.shape == predictions.shape)