from paddle.v2.gradient_machine import GradientMachine

__all__ = ['Optimizer']


class TrainEvent(object):
    """
    The base class for train observe_callback event
    """
    pass


def default_observe_callback(event):
    assert isinstance(event, TrainEvent)


class Optimizer(object):
    def __init__(self, gradient_machine):
        if not isinstance(gradient_machine, GradientMachine):
            raise ValueError()

        self.gradient_machine = gradient_machine

    def train(self,
              train_reader,
              test_reader=None,
              cost=None,
              metrics=None,
              observe_callback=None):
        """
        Training a model.

        :param train_reader: The train reader is a method will return a python
                             iterator, which can return a mini-batch of train
                             data every time.
        :param test_reader: The test reader is same as train reader but return
                            test data. Could be None if there is no test data.
        :param cost: Training costs.
        :param metrics: Training metrics.
        :param observe_callback: The callback will be invoked on each training
                                 step. User could pass a observe_callback to
                                 print/plot training log.
        :return:
        """
        if observe_callback is None:
            observe_callback = default_observe_callback
        raise NotImplementedError()
