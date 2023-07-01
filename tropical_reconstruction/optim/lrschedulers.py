import math
import random
import numpy as np

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
    
    def step(self, multiplicity=1):
        self._lr = self.start/(math.pow(multiplicity,self.scale))


class FeasibilityConeLRScheduler(LRScheduler):
    """ Sets the learning rate to be the minimum of the starting learning rate
    and the conservative feasibility threshold tau
    """
    
    def __init__(self, start=0.01, method="conservative"):
        assert method in ["conservative","random","aggressive"]
        self.method = method
        super().__init__(start=start)

    def step(self, hausdorff_pts = None, grad = None):
        taus = []
        for p,q,e in hausdorff_pts:
            pushforward = grad[:-1]@e + grad[-1]
            tau = 2*np.dot(pushforward, p-q) / np.linalg.norm(pushforward)**2
            taus.append(taus)

        TAU = getattr(self, "_"+self.method)(taus)
        return min(TAU, self.start)

    def _conservative(self, taus: list):
        return min(taus)

    def _random(self, taus: list):
        return random.chioce(taus)

    def _aggressive(self, taus: list):
        return max(taus)
