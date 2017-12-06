import abc


class GroupFusedLasso(metaclass=abc.ABCMeta):
    """
    Base class for the group fused Lasso.
    """

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abc.abstractmethod
    def detect(self, Y, **kwargs):
        """
        Detect the breakpoints in Y.
        """
