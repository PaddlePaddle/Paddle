import collections

import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter

from paddle.proto.ModelConfig_pb2 import ModelConfig
from . import optimizer as v2_optimizer
from . import parameters as v2_parameters
from . import event as v2_event

__all__ = ['ITrainer', 'SGD']


def default_event_handler(event):
    """
    Default event handler. It will print some log and save mode.

    TODO(yuyang18): Complete it!
    :param event:
    :return:
    """
    pass


class ITrainer(object):
    """
    The interface of Trainer. The only exposed method is `train`.
    """

    def train(self,
              train_data_reader,
              topology,
              parameters,
              test_data_reader=None,
              event_handler=None):
        """
        train method.

        :param train_data_reader:
        :param topology:
        :param parameters:
        :param test_data_reader:
        :param event_handler:
        :return:
        """

        raise NotImplementedError()


class SGD(ITrainer):
    def __init__(self, update_equation):
        """
        Simple SGD Trainer.

        :param update_equation: The optimizer object.
        :type update_equation: v2_optimizer.Optimizer
        """
        if not isinstance(update_equation, v2_optimizer.Optimizer):
            raise ValueError("update equation parameter must be "
                             "paddle.v2.optimizer.Optimizer")
        self.__optimizer__ = update_equation

    def train(self,
              train_data_reader,
              topology,
              parameters,
              num_passes=1,
              test_data_reader=None,
              event_handler=None,
              batch_size=32,
              data_types=None):
        """
        Training method. Will train num_passes of input data.

        :param train_data_reader:
        :param topology: Network Topology, a protobuf ModelConfig message.
        :param parameters: The parameter pools.
        :param num_passes: The total train passes.
        :param test_data_reader:
        :param event_handler: Event handler. A method will be invoked when event
                              occurred.
        :type event_handler: (BaseEvent) => None
        :param batch_size: Not important, will be removed after data refactor.
        :param data_types: Not important, will be removed after data refactor.
        :return:
        """
        if event_handler is None:
            event_handler = default_event_handler

        __check_train_args__(**locals())

        gm = api.GradientMachine.createFromConfigProto(
            topology, api.CREATE_MODE_NORMAL, self.__optimizer__.enable_types())
        assert isinstance(gm, api.GradientMachine)
        parameters.append_gradient_machine(gm)

        updater = self.__optimizer__.create_local_updater()
        updater.init(gm)

        gm.start()
        out_args = api.Arguments.createArguments(0)

        data_types_lists = []
        for each in topology.input_layer_names:
            if each not in data_types:
                raise ValueError()
            data_types_lists.append(data_types[each])

        converter = DataProviderConverter(input_types=data_types_lists)

        for pass_id in xrange(num_passes):
            updater.startPass()
            for batch_id, data_batch in enumerate(
                    __data_reader_to_batch__(train_data_reader, batch_size,
                                             topology)):
                pass_type = updater.startBatch(len(data_batch))
                gm.forwardBackward(converter(data_batch), out_args, pass_type)
                for each_param in gm.getParameters():
                    updater.update(each_param)
                # Get cost. We use numpy to calculate total cost for this batch.
                cost_vec = out_args.getSlotValue(0)
                cost_vec = cost_vec.copyToNumpyMat()
                cost = cost_vec.sum() / len(data_batch)
                updater.finishBatch(cost)
                event_handler(
                    v2_event.EndIteration(
                        pass_id=pass_id, batch_id=batch_id, cost=cost))

            updater.finishPass()
        gm.finish()


def __data_reader_to_batch__(reader, batch_size, topology):
    """
    This function is not important, and will be removed when data refactored.
    """

    def input_reorder(func):
        for item in func():
            retv = []
            for __layer_name__ in topology.input_layer_names:
                retv.append(item[__layer_name__])
            yield retv

    return __generator_to_batch__(input_reorder(reader), batch_size=batch_size)


def __generator_to_batch__(generator, batch_size):
    """
    This function is not important, and will be removed when data refactored.
    """
    ret_val = list()
    for each_item in generator:
        ret_val.append(each_item)
        if len(ret_val) == batch_size:
            yield ret_val
            ret_val = list()
    if len(ret_val) != 0:
        yield ret_val


def __check_train_args__(train_data_reader, topology, parameters,
                         test_data_reader, event_handler, **kwargs):
    """
    Check train function's argument types
    """
    if not callable(train_data_reader) or not isinstance(train_data_reader(),
                                                         collections.Iterator):
        raise ValueError('train_data_reader should be a function, '
                         'which can return a iterator')

    if test_data_reader is not None:
        if not callable(test_data_reader) or not isinstance(
                test_data_reader(), collections.Iterator):
            raise ValueError('test_data_reader should be a function, which can '
                             'return a iterator')

    if not isinstance(topology, ModelConfig):
        raise ValueError('topology should be a model config')

    if not isinstance(parameters, v2_parameters.Parameters):
        raise ValueError('parameters should be a parameter pool')

    if not callable(event_handler):
        raise ValueError('event handler should be a function')
