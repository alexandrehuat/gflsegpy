import abc


class GroupFusedLasso(metaclass=abc.ABCMeta):
    """
    Base class for the group fused Lasso.
    """

    @abc.abstractmethod
    def detect(self, Y, **kwargs):
        """
        Detect the breakpoints in Y.
        """
