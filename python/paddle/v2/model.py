import layers as v2_layer
import optimizer as v2_optimizer
import py_paddle.swig_paddle as api
import collections


class Evaluator(object):
    def __init__(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class Model(object):
    """
    :type __gradient_machine__: api.GradientMachine
    :type __updater__: api.ParameterUpdater
    """

    def __init__(self, layers, optimizer, is_local=True):
        if not isinstance(layers, collections.Sequence):
            layers = [layers]  # layers could be many.

        assert is_local, 'Currently, only local mode is supported'

        for each in layers:
            assert isinstance(each, v2_layer.Layer)

        assert isinstance(optimizer, v2_optimizer.Optimizer)

        # Create Proto.
        self.__gradient_machine__ = self.create_gradient_machine(layers,
                                                                 optimizer)

        self.__updater__ = self.create_local_updater(optimizer)

    def rand_parameter(self):
        raise NotImplementedError()

    def save_parameter(self, filename):
        raise NotImplementedError()

    def load_parameter(self, filename):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    def start_pass(self):
        raise NotImplementedError()

    def start_batch(self):
        raise NotImplementedError()

    def finish_batch(self):
        raise NotImplementedError()

    def train(self, data):
        in_args = self.__data_converter(data)
        out_args = api.Arguments.createArguments(0)
        # forward/backward
        # update
        raise NotImplementedError()

    def test(self, data):
        self.__updater__.catchUpWith()
        in_args = self.__data_converter(data)
        out_args = api.Arguments.createArguments(0)
        # forward.
        raise NotImplementedError()

    def make_evaluator(self):
        """

        :return:
        :rtype: Evaluator
        """
        raise NotImplementedError()

    def finish_pass(self):
        raise NotImplementedError()

    def complete(self):
        return self.finish()

    def create_gradient_machine(self, layers, optimizer):
        raise NotImplementedError()

    def create_local_updater(self, optimizer):
        raise NotImplementedError()

    @property
    def __data_converter(self):
        raise NotImplementedError()
