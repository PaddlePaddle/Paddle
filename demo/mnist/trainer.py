import functools
from py_paddle import swig_paddle as api
from py_paddle import DataProviderConverter
from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers import inputs as ipts
import random

__all__ = [
    'RunnerChainItem', 'Runner', 'DeviceChainItem', 'CreateGradientMachine',
    'RandomInitializeParams', 'BasicLocalParameterUpdater', 'network',
    'BasicTrainerDataProvider', 'BasicDataProviderOps',
    'BasicGradientMachineTrainOps', 'Counter', 'BatchEvaluate',
    'BasicTestDataProvider', 'TestOnPassEnd'
]


class NetworkConfig(object):
    def __init__(self):
        pass

    @property
    def input_order(self):
        raise NotImplemented()

    @property
    def input_types(self):
        raise NotImplemented()

    def network_graph(self):
        raise NotImplemented()

    def optimize_graph(self):
        raise NotImplemented()


def network(inputs, **opt_kwargs):
    def __impl__(func):
        functools.wraps(func)

        class NetworkConfigImpl(NetworkConfig):
            def __init__(self):
                NetworkConfig.__init__(self)
                self.__inputs__ = inputs
                self.__network_graph__ = None
                self.__optimize_graph__ = None

            @property
            def input_order(self):
                return inputs.keys()

            @property
            def input_types(self):
                return self.__inputs__

            def network_graph(self):
                if self.__network_graph__ is None:

                    def __network_graph_func__():
                        kwargs = dict()
                        lst = list()
                        for k in inputs:
                            v = inputs[k]
                            data = data_layer(name=k, size=v.dim)
                            kwargs[k] = data
                            lst.append(data)
                        ipts(*lst)
                        rst = func(**kwargs)
                        if not isinstance(rst, tuple):
                            rst = (rst, )
                        outputs(*rst)

                    self.__network_graph__ = parse_network_config(
                        __network_graph_func__)
                return self.__network_graph__

            def optimize_graph(self):
                if self.__optimize_graph__ is None:

                    def __optimize_graph_func__():
                        settings(**opt_kwargs)

                    self.__optimize_graph__ = parse_optimizer_config(
                        __optimize_graph_func__)
                return self.__optimize_graph__

        return NetworkConfigImpl

    return __impl__


class RunnerChainItem(object):
    def __init__(self):
        pass

    def initialize(self, context, next_callback):
        next_callback(context)

    def finalize(self, context, next_callback):
        next_callback(context)

    def on_pass_begin(self, context, next_callback):
        next_callback(context)

    def on_pass_end(self, context, next_callback):
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        return next_callback(context)

    def on_batch_end(self, context, next_callback):
        return next_callback(context)


def default_next_callback(*args, **kwargs):
    return False


class RunnerContext(object):
    pass


class RunnerSection(object):
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        self.runner.initialize()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runner.finalize(self.runner.context)


class Runner(object):
    def __init__(self):
        self.chains = []

        self.begin_pass = None
        self.end_pass = None
        self.begin_batch = None
        self.end_batch = None
        self.finalize = None

        self.context = RunnerContext()
        self.context.runner = self

    def add_chain_item(self, item):
        assert isinstance(item, RunnerChainItem)
        self.chains.append(item)

    def initialize(self, parent=None):
        if None not in [
                self.begin_pass, self.end_pass, self.begin_batch,
                self.end_batch, self.finalize
        ]:
            return False
        else:
            assert len(self.chains) != 0
            actual_init = default_next_callback
            self.begin_pass = default_next_callback
            self.end_pass = default_next_callback
            self.begin_batch = default_next_callback
            self.end_batch = default_next_callback
            self.finalize = default_next_callback

            for chain in reversed(self.chains):
                assert isinstance(chain, RunnerChainItem)
                actual_init = functools.partial(
                    chain.initialize, next_callback=actual_init)
                self.begin_pass = functools.partial(
                    chain.on_pass_begin, next_callback=self.begin_pass)
                self.end_pass = functools.partial(
                    chain.on_pass_end, next_callback=self.end_pass)
                self.begin_batch = functools.partial(
                    chain.on_batch_begin, next_callback=self.begin_batch)
                self.end_batch = functools.partial(
                    chain.on_batch_end, next_callback=self.end_batch)
                self.finalize = functools.partial(
                    chain.finalize, next_callback=self.finalize)

            if parent is not None:
                self.context.parent = parent

            actual_init(self.context)
            return True

    def run_one_pass(self, parent=None):
        if parent is not None:
            self.context.parent = parent

        self.begin_pass(self.context)
        exit_flag = False
        while not exit_flag:
            exit_flag = self.begin_batch(self.context)
            if exit_flag:
                break
            exit_flag = self.end_batch(self.context)
        self.end_pass(self.context)

    def use(self):
        return RunnerSection(self)


class DeviceChainItem(RunnerChainItem):
    def __init__(self, use_gpu=False, device_count=4):
        RunnerChainItem.__init__(self)
        self.use_gpu = use_gpu
        self.device_count = device_count

    def initialize(self, context, next_callback):
        api.initPaddle('--use_gpu=%s' % repr(self.use_gpu),
                       '--trainer_count=%d' % self.device_count)
        next_callback(context)


class CreateGradientMachine(RunnerChainItem):
    def __init__(self, network):
        RunnerChainItem.__init__(self)
        assert isinstance(network, NetworkConfig)
        self.__network__ = network

    def initialize(self, context, next_callback):
        # get enable_types for each optimizer.
        # enable_types = [value, gradient, momentum, etc]
        # For each optimizer(SGD, Adam), GradientMachine should enable different
        # buffers.
        opt_config = api.OptimizationConfig.createFromProto(
            self.__network__.optimize_graph())
        _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
        enable_types = _temp_optimizer_.getParameterTypes()

        # Create Simple Gradient Machine.
        context.gradient_machine = api.GradientMachine.createFromConfigProto(
            self.__network__.network_graph(), api.CREATE_MODE_NORMAL,
            enable_types)
        # This type check is not useful. Only enable type hint in IDE.
        # Such as PyCharm
        assert isinstance(context.gradient_machine, api.GradientMachine)

        next_callback(context)
        context.gradient_machine.start()

    def finalize(self, context, next_callback):
        context.gradient_machine.finish()
        next_callback(context)


class RandomInitializeParams(RunnerChainItem):
    def __init__(self):
        RunnerChainItem.__init__(self)

    def initialize(self, context, next_callback):
        assert hasattr(context, 'gradient_machine') and isinstance(
            context.gradient_machine, api.GradientMachine)
        context.gradient_machine.randParameters()
        next_callback(context)


class BasicLocalParameterUpdaterOps(RunnerChainItem):
    def __init__(self,
                 updater_name='updater',
                 batch_size_name='current_batch_size',
                 cost_name='current_cost'):
        RunnerChainItem.__init__(self)
        self.__updater_name__ = updater_name
        self.__batch_size_name__ = batch_size_name
        self.__cost_name__ = cost_name

    def on_pass_begin(self, context, next_callback):
        self.__get_updater__(context).startPass()
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        self.__get_updater__(context).startBatch(
            self.__get_batch_size__(context))
        return next_callback(context)

    def on_batch_end(self, context, next_callback):
        exit_flag = next_callback(context)
        if not exit_flag:
            self.__get_updater__(context).finishBatch(
                self.__get_cost__(context))
        return exit_flag

    def on_pass_end(self, context, next_callback):
        self.__get_updater__(context).finishPass()
        next_callback(context)

    def __get_updater__(self, context):
        ret = getattr(context, self.__updater_name__, None)
        assert isinstance(ret, api.ParameterUpdater)
        return ret

    def __get_batch_size__(self, context):
        return getattr(context, self.__batch_size_name__, None)

    def __get_cost__(self, context):
        return getattr(context, self.__cost_name__, None)


class BasicLocalParameterUpdater(BasicLocalParameterUpdaterOps):
    def __init__(self, network):
        BasicLocalParameterUpdaterOps.__init__(self)
        self.__network__ = network

    def initialize(self, context, next_callback):
        assert hasattr(context, 'gradient_machine') and isinstance(
            context.gradient_machine, api.GradientMachine)

        # Create Local Updater. Local means not run in cluster.
        # For a cluster training, here we can change to createRemoteUpdater
        # in future.
        opt_config = api.OptimizationConfig.createFromProto(
            self.__network__.optimize_graph())
        context.updater = api.ParameterUpdater.createLocalUpdater(opt_config)
        assert isinstance(context.updater, api.ParameterUpdater)
        context.updater.init(context.gradient_machine)
        context.updater_callback = context.updater.update
        next_callback(context)


class BasicGradientMachineTrainOps(RunnerChainItem):
    def __init__(self):
        RunnerChainItem.__init__(self)
        self.__out_args__ = api.Arguments.createArguments(0)

    def on_batch_begin(self, context, next_callback):
        # forwardBackward is a shortcut for forward and backward.
        # It is sometimes faster than invoke forward/backward separately,
        # because in GradientMachine, it may be async.
        context.gradient_machine.forwardBackward(
            context.in_args, self.__out_args__, api.PASS_TRAIN)

        assert isinstance(self.__out_args__, api.Arguments)

        for each_param in context.gradient_machine.getParameters():
            context.updater_callback(each_param)

        context.current_cost = self.__out_args__.sumCosts(
        ) / context.current_batch_size

        return next_callback(context)


class Counter(RunnerChainItem):
    def __init__(self):
        RunnerChainItem.__init__(self)

    def initialize(self, context, next_callback):
        context.current_pass_id = 0
        context.current_batch_id = 0
        next_callback(context)

    def on_batch_end(self, context, next_callback):
        ret = next_callback(context)
        context.current_batch_id += 1
        return ret

    def on_pass_end(self, context, next_callback):
        next_callback(context)
        context.current_pass_id += 1


class BaseEvaluate(RunnerChainItem):
    def __init__(self, prefix=None):
        RunnerChainItem.__init__(self)
        self.__evaluator__ = None
        if prefix is None:
            prefix = ''
        self.__prefix__ = prefix

    def initialize(self, context, next_callback):
        next_callback(context)
        assert isinstance(context.gradient_machine, api.GradientMachine)
        self.__evaluator__ = context.gradient_machine.makeEvaluator()

    def finalize(self, context, next_callback):
        next_callback(context)
        self.__evaluator__ = None


class BatchEvaluate(BaseEvaluate):
    def __init__(self, prefix=None):
        BaseEvaluate.__init__(self, prefix)

    def on_batch_end(self, context, next_callback):
        assert isinstance(context.gradient_machine, api.GradientMachine)
        self.__evaluator__.start()
        context.gradient_machine.eval(self.__evaluator__)
        retv = next_callback(context)
        print '%sPass=%d, Batch=%d Cost=%f, Eval:' % (self.__prefix__,
                                                      context.current_pass_id,
                                                      context.current_batch_id,
                                                      context.current_cost), \
            self.__evaluator__
        self.__evaluator__.finish()
        return retv


class PassEvaluate(BaseEvaluate):
    def __init__(self, prefix=None):
        BaseEvaluate.__init__(self, prefix)

    def on_pass_begin(self, context, next_callback):
        next_callback(context)
        self.__evaluator__.start()

    def on_batch_end(self, context, next_callback):
        retv = next_callback(context)
        context.gradient_machine.eval(self.__evaluator__)
        return retv

    def on_pass_end(self, context, next_callback):
        next_callback(context)
        print '%sPass=%d Eval:' % (self.__prefix__, context.current_pass_id), \
            self.__evaluator__
        self.__evaluator__.finish()


class BasicGradientMachineTestOps(RunnerChainItem):
    def __init__(self):
        RunnerChainItem.__init__(self)
        self.__out_args__ = api.Arguments.createArguments(0)

    def on_pass_begin(self, context, next_callback):
        context.updater.apply()
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        context.gradient_machine.forward(context.in_args, self.__out_args__,
                                         api.PASS_TEST)
        return next_callback(context)

    def on_pass_end(self, context, next_callback):
        context.updater.restore()
        next_callback(context)


class InheritGradientMachineUpdater(RunnerChainItem):
    def __init__(self):
        RunnerChainItem.__init__(self)

    def initialize(self, context, next_callback):
        if context.parent is not None:
            context.gradient_machine = context.parent.gradient_machine
            context.updater = context.parent.updater
        next_callback(context)

    def on_pass_begin(self, context, next_callback):
        if context.parent is not None:
            context.current_pass_id = context.parent.current_pass_id
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        if context.parent is not None:
            context.current_batch_id = context.parent.current_batch_id
        return next_callback(context)


class TestOnPassEnd(RunnerChainItem):
    def __init__(self, **kwargs):
        RunnerChainItem.__init__(self)
        self.__test_runner__ = Runner()
        self.__test_runner__.add_chain_item(InheritGradientMachineUpdater())
        self.__test_runner__.add_chain_item(BasicTestDataProvider(**kwargs))
        self.__test_runner__.add_chain_item(BasicGradientMachineTestOps())
        self.__test_runner__.add_chain_item(PassEvaluate(prefix='Test: '))

    def initialize(self, context, next_callback):
        next_callback(context)
        self.__test_runner__.initialize(context)

    def on_pass_end(self, context, next_callback):
        self.__test_runner__.run_one_pass(parent=context)
        next_callback(context)


class DataProvider(object):
    __slots__ = ['__init__', 'reset', 'next']

    def __init__(self):
        pass

    def reset(self):
        raise NotImplemented()

    def next(self):
        raise NotImplemented()


class NaiveDataProvider(DataProvider):
    def __init__(self, provider, input_types, batch_size, should_shuffle=True):
        DataProvider.__init__(self)
        self.__converter__ = DataProviderConverter(input_types)
        if provider.should_shuffle is None:
            provider.should_shuffle = should_shuffle

        self.__provider__ = provider
        self.__pool__ = []
        self.__batch_size__ = batch_size
        self.__idx__ = 0

    def reset(self):
        def __to_pool__():
            for filename in self.__provider__.file_list:
                for item in self.__provider__.generator(self.__provider__,
                                                        filename):
                    yield item

        self.__pool__ = list(__to_pool__())
        if self.__provider__.should_shuffle:
            random.shuffle(self.__pool__)

        self.__idx__ = 0

    def next(self):
        if self.__idx__ < len(self.__pool__):
            end = min(self.__idx__ + self.__batch_size__, len(self.__pool__))
            begin = self.__idx__
            self.__idx__ = end
            return self.__converter__(self.__pool__[begin:end]), end - begin
        else:
            raise StopIteration


class BasicDataProviderOps(RunnerChainItem):
    def __init__(self, provider_name='data_provider'):
        RunnerChainItem.__init__(self)
        self.__provider_name__ = provider_name

    def __get_provider__(self, context):
        ret = getattr(context, self.__provider_name__, None)
        assert isinstance(ret, DataProvider)
        return ret

    def on_pass_begin(self, context, next_callback):
        dp = self.__get_provider__(context)
        dp.reset()
        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        dp = self.__get_provider__(context)
        try:
            context.in_args, context.current_batch_size = next(dp)
            return next_callback(context)
        except StopIteration:
            return True

    def on_batch_end(self, context, next_callback):
        context.in_args = None
        context.current_batch_size = None
        return next_callback(context)


def data_provider_creator(is_train):
    class __cls__(BasicDataProviderOps):
        def __init__(self, network, method, file_list, batch_size, **kwargs):
            BasicDataProviderOps.__init__(self)
            assert isinstance(network, NetworkConfig)
            self.__dataprovider__ = method(
                file_list=file_list,
                input_order=network.input_order,
                is_train=is_train,
                **kwargs)
            self.__input_types__ = []
            for data_layer_name in network.input_order:
                self.__input_types__.append(network.input_types[
                    data_layer_name])
            self.__batch_size__ = batch_size

        def initialize(self, context, next_callback):
            context.data_provider = NaiveDataProvider(
                provider=self.__dataprovider__,
                input_types=self.__input_types__,
                batch_size=self.__batch_size__,
                should_shuffle=True if is_train else False)
            next_callback(context)

    return __cls__


BasicTrainerDataProvider = data_provider_creator(True)
BasicTestDataProvider = data_provider_creator(False)
