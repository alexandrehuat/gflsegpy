import abc


class GroupFusedLasso(metaclass=abc.ABCMeta):
    """
    Base class for the group fused Lasso.
    """

    @abc.abstractmethod
    def detect(self, X, **kwargs):
        """
        Detect the breakpoints in X.
        :param X:
        :param kwargs:
        :return:
        """
