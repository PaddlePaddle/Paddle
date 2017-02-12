from paddle.v2.evaluator import Evaluator


class GradientMachine(object):
    def __init__(self, evaluator):
        if not isinstance(evaluator, Evaluator):
            raise ValueError()

        self.evaluator = evaluator

    def forward_backward(self, data_batch, cost=None, metrics=None):
        """
        Apply a forward-backward operations to a neural network.

        The arguments of this method basically as same as Evaluator.forward
        See Evaluator.forward's documentation for details.
        """
        raise NotImplementedError()

    def gradient(self, parameter_name):
        """
        Get gradient of one parameter.

        :param parameter_name:
        :return:
        """
        raise NotImplementedError()
