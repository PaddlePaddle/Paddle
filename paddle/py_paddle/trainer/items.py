from .data_providers import *
from .base import RunnerItem, Runner
from .network import NetworkConfig
from .base_items import *
from .. import swig_paddle as api

__all__ = [
    'set_device', 'std_random_init_params', 'CreateGradientMachine',
    'BasicLocalParameterUpdaterOps', 'BasicLocalParameterUpdater', 'Counter',
    'BatchEvaluate', 'PassEvaluate', 'BasicGradientMachineTrainOps',
    'BasicGradientMachineTestOps', 'InheritGradientMachineUpdater',
    'TestOnPassEnd', 'BasicPaddleTrainerDataProvider', 'BasicDataProviderOps',
    'BasicPaddleTestDataProvider', 'SaveParamsOnPassEnd', 'BaseObserveItem'
]


@init_runner_item()
def set_device(use_gpu, device_count, **kwargs):
    """
    Set Device Of PaddlePaddle. It will invoke api.initPaddle, so must
    be the first RunnerItem.

    @TODO(yuyang18): Check current Paddle compiled with CUDA support and
                     Check the max device count in system.
    :param use_gpu: True if using GPU.
    :param device_count: CPU cores or GPU cards count.
    :param kwargs: Not used. But just added for capacity
    :return: None
    """
    api.initPaddle('--use_gpu=%s' % repr(use_gpu),
                   '--trainer_count=%d' % device_count)


@init_runner_item()
def std_random_init_params(context, **kwargs):
    """

    :param context:
    :type context: ContextWrapper
    :return:
    """
    ContextWrapper(context).gradient_machine().randParameters()


class CreateGradientMachine(BaseRunnerItem):
    def __init__(self, network, local=True):
        RunnerItem.__init__(self)
        assert isinstance(network, NetworkConfig)
        self.__network__ = network
        self.__local__ = local

    def initialize(self, context, next_callback):
        """

        :param context:
        :type context: ContextWrapper
        :param next_callback:
        :return:
        """
        self.store_context(context)
        # get enable_types for each optimizer.
        # enable_types = [value, gradient, momentum, etc]
        # For each optimizer(SGD, Adam), GradientMachine should enable different
        # buffers.
        opt_config = api.OptimizationConfig.createFromProto(
            self.__network__.optimize_graph())
        _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
        enable_types = _temp_optimizer_.getParameterTypes()

        graph = self.__network__.network_graph()

        if self.__local__:
            for param in graph.parameters:
                if param.HasField("sparse_remote_update"):
                    param.sparse_remote_update = False

        self.context.set_gradient_machine(
            api.GradientMachine.createFromConfigProto(
                graph, api.CREATE_MODE_NORMAL, enable_types))
        next_callback(context)
        self.context.gradient_machine().start()

    def finalize(self, next_callback):
        self.context.gradient_machine().finish()
        next_callback()


class BasicLocalParameterUpdaterOps(BaseRunnerItem):
    """
    Basic Operations for local parameter updater.

    Invoke startPass/finishPass etc.
    """

    def __init__(self):
        super(BasicLocalParameterUpdaterOps, self).__init__()

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)

    def on_pass_begin(self, next_callback):
        self.context.updater().startPass()
        next_callback()

    def on_batch_begin(self, next_callback):
        self.context.updater().startBatch(self.context.batch_size())
        return next_callback()

    def on_batch_end(self, next_callback):
        exit_flag = next_callback()
        if not exit_flag:
            self.context.updater().finishBatch(self.context.cost())
        return exit_flag

    def on_pass_end(self, next_callback):
        self.context.updater().finishPass()
        next_callback()


class BasicLocalParameterUpdater(BasicLocalParameterUpdaterOps):
    """
    Create a basic local parameter updater.
    """

    def __init__(self, network):
        super(BasicLocalParameterUpdater, self).__init__()
        self.__network__ = network

    def initialize(self, context, next_callback):
        self.store_context(context)
        # Create Local Updater. Local means not run in cluster.
        # For a cluster training, here we can change to createRemoteUpdater
        # in future.
        opt_config = api.OptimizationConfig.createFromProto(
            self.__network__.optimize_graph())
        self.context.set_updater(
            api.ParameterUpdater.createLocalUpdater(opt_config))
        self.context.updater().init(self.context.gradient_machine())
        self.context.set_updater_callback(self.context.updater().update)
        next_callback(context)


class Counter(BaseRunnerItem):
    def __init__(self):
        super(Counter, self).__init__()

    def initialize(self, context, next_callback):
        self.store_context(context)
        self.context.reset_batch_id()
        self.context.reset_pass_id()
        next_callback(context)

    def on_batch_end(self, next_callback):
        ret = next_callback()
        self.context.increase_batch_id()
        return ret

    def on_pass_end(self, next_callback):
        next_callback()
        self.context.increase_pass_id()


class BaseEvaluate(BaseRunnerItem):
    """
    Base Evaluate Item. Just create a evaluator, it can be used in derived
    class. It will print some stats log during run_one_pass.

    :type __evaluator__: api.Evaluator
    :param prefix: The prefix for stats log.
    :type prefix: basestring
    :type __prefix__: basestring
    """

    def __init__(self, prefix=None):
        super(BaseEvaluate, self).__init__()
        self.__evaluator__ = None
        if prefix is None:
            prefix = ''
        self.__prefix__ = prefix

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)
        self.__evaluator__ = self.context.gradient_machine().makeEvaluator()

    def finalize(self, next_callback):
        next_callback()
        self.__evaluator__ = None


class BatchEvaluate(BaseEvaluate):
    """
    Print stats log on each batch end.
    """

    def __init__(self, prefix=None):
        BaseEvaluate.__init__(self, prefix)

    def on_batch_end(self, next_callback):
        self.__evaluator__.start()
        self.context.gradient_machine().eval(self.__evaluator__)
        retv = next_callback()
        print '%sPass=%d, Batch=%d Cost=%f, Eval:' % (self.__prefix__,
                                                      self.context.pass_id(),
                                                      self.context.batch_id(),
                                                      self.context.cost()), \
            self.__evaluator__
        self.__evaluator__.finish()
        return retv


class PassEvaluate(BaseEvaluate):
    """
    Print stats log on each pass end.
    """

    def __init__(self, prefix=None):
        super(PassEvaluate, self).__init__(prefix=prefix)

    def on_pass_begin(self, next_callback):
        next_callback()
        self.__evaluator__.start()

    def on_batch_end(self, next_callback):
        retv = next_callback()
        self.context.gradient_machine().eval(self.__evaluator__)
        return retv

    def on_pass_end(self, next_callback):
        next_callback()
        print '%sPass=%d Eval:' % (self.__prefix__, self.context.pass_id()), \
            self.__evaluator__
        self.__evaluator__.finish()


class BasicGradientMachineTrainOps(BaseRunnerItem):
    """
    Forward/backward a gradient machine.

    :type __out_args__: api.Arguments
    """

    def __init__(self):
        super(BasicGradientMachineTrainOps, self).__init__()
        self.__out_args__ = api.Arguments.createArguments(0)

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)

    def on_batch_begin(self, next_callback):
        # forwardBackward is a shortcut for forward and backward.
        # It is sometimes faster than invoke forward/backward separately,
        # because in GradientMachine, it may be async.

        self.context.real_context.gradient_machine.forwardBackward(
            self.context.in_args(), self.__out_args__, api.PASS_TRAIN)

        for each_param in self.context.gradient_machine().getParameters():
            self.context.real_context.updater_callback(each_param)

        self.context.set_cost(self.__out_args__.sumCosts() /
                              self.context.batch_size())
        return next_callback()


class BasicGradientMachineTestOps(BaseRunnerItem):
    def __init__(self):
        super(BasicGradientMachineTestOps, self).__init__()
        self.__out_args__ = api.Arguments.createArguments(0)

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)

    def on_pass_begin(self, next_callback):
        self.context.updater().apply()
        next_callback()

    def on_batch_begin(self, next_callback):
        self.context.gradient_machine().forward(
            self.context.in_args(), self.__out_args__, api.PASS_TEST)
        return next_callback()

    def on_pass_end(self, next_callback):
        self.context.updater().restore()
        next_callback()


class InheritGradientMachineUpdater(RunnerItem):
    def __init__(self):
        RunnerItem.__init__(self)
        self.context = None

    def initialize(self, context, next_callback):
        if context.parent is not None:  # inherit from parent.
            context.gradient_machine = context.parent.gradient_machine
            context.updater = context.parent.updater
        self.context = context
        next_callback(context)

    def on_pass_begin(self, next_callback):
        if self.context.parent is not None:
            self.context.current_pass_id = self.context.parent.current_pass_id
        next_callback()

    def on_batch_begin(self, next_callback):
        if self.context.parent is not None:
            self.context.current_batch_id = self.context.parent.current_batch_id
        return next_callback()


class BasicDataProviderOps(BaseRunnerItem):
    """
    :type __provider__: DataProvider
    """

    def __init__(self):
        super(BasicDataProviderOps, self).__init__()
        self.__provider__ = None

    def on_pass_begin(self, next_callback):
        self.__provider__.reset()
        next_callback()

    def on_batch_begin(self, next_callback):
        try:
            in_args, batch_size = next(self.__provider__)
            self.context.set_in_args(in_args)
            self.context.set_batch_size(batch_size)
            return next_callback()
        except StopIteration:
            return True


def data_provider_creator(is_train):
    class __cls__(BasicDataProviderOps):
        def __init__(self, network, method, file_list, batch_size, **kwargs):
            super(__cls__, self).__init__()
            self.__dataprovider__ = method(
                file_list=file_list,
                input_order=network.input_order(),
                is_train=is_train,
                **kwargs)
            self.__input_types__ = []
            for data_layer_name in network.input_order():
                self.__input_types__.append(network.input_types()[
                    data_layer_name])
            self.__batch_size__ = batch_size

        def initialize(self, context, next_callback):
            self.store_context(context)
            self.__provider__ = NaiveDataProvider(
                provider=self.__dataprovider__,
                input_types=self.__input_types__,
                batch_size=self.__batch_size__,
                should_shuffle=True if is_train else False)
            next_callback(context)

    return __cls__


BasicPaddleTrainerDataProvider = data_provider_creator(True)
BasicPaddleTestDataProvider = data_provider_creator(False)


class TestOnPassEnd(RunnerItem):
    def __init__(self, **kwargs):
        RunnerItem.__init__(self)
        self.__test_runner__ = Runner()
        self.__test_runner__.add_item(InheritGradientMachineUpdater())
        self.__test_runner__.add_item(BasicPaddleTestDataProvider(**kwargs))
        self.__test_runner__.add_item(BasicGradientMachineTestOps())
        self.__test_runner__.add_item(PassEvaluate(prefix='Test: '))

    def initialize(self, context, next_callback):
        next_callback(context)
        self.__test_runner__.__initialize__(context)

    def on_pass_end(self, next_callback):
        self.__test_runner__.run_one_pass()
        next_callback()


class SaveParamsOnPassEnd(BaseRunnerItem):
    def __init__(self):
        super(SaveParamsOnPassEnd, self).__init__()

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)

    def on_pass_end(self, next_callback):
        self.context.updater().catchUpWith()
        params = self.context.gradient_machine().getParameters()
        for param in params:
            param.save(param.getName())

        next_callback()


class BaseObserveItem(BaseRunnerItem):
    def __init__(self, batch_end_callback=None, pass_end_callback=None):
        super(BaseObserveItem, self).__init__()
        self.__batch_end__ = batch_end_callback
        self.__pass_end__ = pass_end_callback

    def initialize(self, context, next_callback):
        self.store_context(context)
        next_callback(context)

    def on_batch_end(self, next_callback):
        r = next_callback()
        if self.__batch_end__ is not None:
            self.__batch_end__(self.context)
        return r

    def on_pass_end(self, next_callback):
        next_callback()
        if self.__pass_end__ is not None:
            self.__pass_end__(self.context)
