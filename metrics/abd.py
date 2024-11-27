from ignite.metrics import MeanAbsoluteError

class AbsoluteBinDistance(MeanAbsoluteError):
    def __init__(self):
        super().__init__()

    def update(self, output):
        y_pred, y = output
        
        y_pred = y_pred.argmax(dim=-1)        
        super(AbsoluteBinDistance, self).update((y_pred, y))