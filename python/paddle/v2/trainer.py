import collections
from paddle.proto.ModelConfig_pb2 import ModelConfig
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
from . import parameters as v2_parameters
import numpy
import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter

__all__ = ['ITrainer', 'SGDTrainer', 'CompleteTrainOneBatch', 'BaseEvent']


class BaseEvent(object):
    """
    Just a marker class
    """
    pass


class CompleteTrainOneBatch(BaseEvent):
    """
    Event On One Batch Training Complete.
    """

    def __init__(self, pass_id, batch_id, cost, parameters):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost
        self.parameters = parameters


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


class LazyParameterPool(v2_parameters.IParameterPool):
    """
    Lazy Parameter Pool stores a reference to GradientMachine. User could invoke
    `get_parameter` if needed, but the operation is lazy. It means the parameter
    will only fetched from GPU or Parameter Server if `get_parameter` is
    invoked. Also, set flag = writable will make a extra host2device copy after
    reading/modifying parameter.

    This class is not exposed to User. User should treat this class as a normal
    IParameterPool.

    See IParameterPool for usage documentation.

    :type __gradient_machine__: api.GradientMachine
    """

    def get_parameter(self, name, flag=v2_parameters.ParameterFlag.READ_WRITE):
        param = filter(lambda x: x.getName() == name,
                       self.__gradient_machine__.getParameters())
        if len(param) == 0:
            raise ValueError("Cannot found parameter with name %s" % name)
        elif len(param) > 1:
            raise RuntimeError("Unexpected branch")
        else:
            conf = param[0].getConfig().toProto()
            param = param[0].getBuf(api.PARAMETER_VALUE)
            assert isinstance(param, api.Vector)
            assert isinstance(conf, ParameterConfig)

        shape = map(int, conf.dims)
        if api.isUsingGpu():
            arr = param.copyToNumpyArray().reshape(shape)
            if flag & v2_parameters.ParameterFlag.WRITE_ONLY:
                self.need_copy = True
                self.arrays[name] = arr
        else:
            arr = param.toNumpyArrayInplace().reshape(shape)
        return arr

    def get_names(self):
        return [
            param.getName()
            for param in self.__gradient_machine__.getParameters()
        ]

    def __init__(self, gradient_machine):
        self.__gradient_machine__ = gradient_machine
        self.need_copy = False
        self.arrays = dict()


class CustomizeUpdateEquation(object):
    def __init__(self, callback):
        self.__callback__ = callback
        if self.__callback__.func_code.co_argcount < 2:
            raise ValueError(
                "The update equation at least should contain 2 arguments, "
                "first is value, second is gradient")

        self.local_params_count = self.__callback__.func_code.co_argcount - 2
        self.local_params = dict()

    def enable_types(self):
        return [api.PARAMETER_VALUE, api.PARAMETER_GRADIENT]

    def init(self, gradient_machine):
        assert isinstance(gradient_machine, api.GradientMachine)
        for param in gradient_machine.getParameters():
            conf = param.getConfig().toProto()
            shape = map(int, conf.dims)
            self.local_params[conf.name] = []
            for _ in xrange(self.local_params_count):
                self.local_params[conf.name].append(
                    numpy.zeros(
                        shape=shape, dtype='float32'))

    def create_local_updater(self):
        return self

    def startPass(self):
        pass

    def finishPass(self):
        pass

    def startBatch(self, batch_size):
        return api.PASS_TRAIN

    def finishBatch(self, cost):
        pass

    def update(self, param):
        conf = param.getConfig().toProto()
        shape = map(int, conf.dims)
        if not api.isUsingGpu():
            v = param.getBuf(api.PARAMETER_VALUE).toNumpyArrayInplace().reshape(
                shape)
            g = param.getBuf(api.PARAMETER_GRADIENT).toNumpyArrayInplace(
            ).reshape(shape)

        else:
            v = param.getBuf(api.PARAMETER_VALUE).copyToNumpyArray().reshape(
                shape)
            g = param.getBuf(api.PARAMETER_GRADIENT).copyToNumpyArray().reshape(
                shape)

        args = [v, g]
        for arg in self.local_params[conf.name]:
            args.append(arg)
        self.__callback__(*args)

        if api.isUsingGpu():
            param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(v.flatten(
            ).astype('float32'))
            # discard gradient changed.


class SGDTrainer(ITrainer):
    def __init__(self, update_equation):
        """
        Simple SGD Trainer.

        :param update_equation: Maybe we should give a DSL for update equation?
        """
        if callable(update_equation):
            update_equation = CustomizeUpdateEquation(update_equation)

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
        __copy_parameter_from_pool__(gm, parameters)

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
                pool = LazyParameterPool(gradient_machine=gm)
                event_handler(
                    CompleteTrainOneBatch(
                        pass_id=pass_id,
                        batch_id=batch_id,
                        cost=cost,
                        parameters=pool))

                if pool.need_copy:
                    __copy_parameter_from_lazy_pool__(gm, pool)

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


def __copy_parameter_from_lazy_pool__(gm, pool):
    assert isinstance(pool, LazyParameterPool)
    for each_param_name in pool.arrays.keys():
        param = filter(lambda x: x.getName() == each_param_name,
                       gm.getParameters())
        assert len(param) == 1
        param = param[0]
        param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(pool.arrays[
            each_param_name].flatten().astype('float32'))


def __copy_parameter_from_pool__(gm, pool):
    """

    :param gm:
    :type gm: api.GradientMachine
    :param pool:
    :type pool: v2_parameters.IParameterPool
    :return:
    """
    assert isinstance(pool, v2_parameters.IParameterPool)
    for each_param in gm.getParameters():
        name = each_param.getName()
        param = pool.get_parameter(name, v2_parameters.ParameterFlag.READ_ONLY)
        each_param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(param.flatten(
        ).astype('float32'))


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

    if not isinstance(parameters, v2_parameters.IParameterPool):
        raise ValueError('parameters should be a parameter pool')

    if not callable(event_handler):
        raise ValueError('event handler should be a function')
