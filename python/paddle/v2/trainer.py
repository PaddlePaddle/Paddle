import collections
from paddle.proto.ModelConfig_pb2 import ModelConfig
import paddle.v2.parameters
import paddle.v2.optimizer
import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter

__all__ = ['ITrainer', 'SGDTrainer', 'CompleteTrainOneBatch', 'BaseEvent']


class BaseEvent(object):
    """
    Just a marker class
    """
    pass


class CompleteTrainOneBatch(BaseEvent):
    def __init__(self, pass_id, batch_id, cost):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost


def default_event_handler(event):
    pass


class ITrainer(object):
    def train(self,
              train_data_reader,
              topology,
              parameters,
              test_data_reader=None,
              event_handler=None):
        raise NotImplementedError()


class SGDTrainer(ITrainer):
    def __init__(self, update_equation):
        if not isinstance(update_equation, paddle.v2.optimizer.Optimizer):
            raise ValueError()

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
        if event_handler is None:
            event_handler = default_event_handler

        __check_train_args__(**locals())

        gm = api.GradientMachine.createFromConfigProto(
            topology, api.CREATE_MODE_NORMAL, self.__optimizer__.enable_types())
        assert isinstance(gm, api.GradientMachine)
        __copy_parameter_from_pool__(gm, parameters)

        updater = self.__optimizer__.create_local_updater()
        assert isinstance(updater, api.ParameterUpdater)
        updater.init(gm)

        data_types_lists = []
        for each in topology.input_layer_names:
            if each not in data_types:
                raise ValueError()
            data_types_lists.append(data_types[each])

        converter = DataProviderConverter(input_types=data_types_lists)

        def input_reorder(func):
            for item in func():
                retv = []
                for __layer_name__ in topology.input_layer_names:
                    retv.append(item[__layer_name__])
                yield retv

        gm.start()

        out_args = api.Arguments.createArguments(0)
        for pass_id in xrange(num_passes):
            updater.startPass()
            for batch_id, data_batch in enumerate(
                    __generator_to_batch__(
                        input_reorder(train_data_reader),
                        batch_size=batch_size)):
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
                    CompleteTrainOneBatch(
                        pass_id=pass_id, batch_id=batch_id, cost=cost))

            updater.finishPass()
        gm.finish()


def __generator_to_batch__(generator, batch_size):
    ret_val = list()
    for each_item in generator:
        ret_val.append(each_item)
        if len(ret_val) == batch_size:
            yield ret_val
            ret_val = list()
    if len(ret_val) != 0:
        yield ret_val


def __copy_parameter_from_pool__(gm, pool):
    """

    :param gm:
    :type gm: api.GradientMachine
    :param pool:
    :type pool: paddle.v2.parameters.IParameterPool
    :return:
    """
    assert isinstance(pool, paddle.v2.parameters.IParameterPool)
    for each_param in gm.getParameters():
        name = each_param.getName()
        param = pool.get_parameter(name,
                                   paddle.v2.parameters.ParameterFlag.READ_ONLY)
        each_param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(param.flatten(
        ).astype('float32'))


def __check_train_args__(train_data_reader, topology, parameters,
                         test_data_reader, event_handler, **kwargs):
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

    if not isinstance(parameters, paddle.v2.parameters.IParameterPool):
        raise ValueError('parameters should be a parameter pool')

    if not callable(event_handler):
        raise ValueError('event handler should be a function')
