import math

class LRScheduler:
    """ Base LR Scheduler (constant scheduler)
    """
    def __init__(self, start=0.01):
        self.start = start
        self._lr = start

    @property
    def lr(self):
        return self._lr

    def step(self, *args):
        pass


class MultiplicityLRScheduler(LRScheduler):
    """ Reduces learning rate based on approximate multiplicity of
    hausdorff distance.
    """

    def __init__(self, start=0.01, scale=1):
        self.scale = scale
        super().__init__(start=start)
    
    def step(self, multiplicity: int):
        self._lr = self.start/(math.pow(multiplicity,self.scale))
