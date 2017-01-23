import traceback

import paddle
import py_paddle.swig_paddle as api

import default_decorators as default_decorators
import optimizer_types as optimizer_types
from logger import logger

try:
    from paddle.proto.TrainerConfig_pb2 import TrainerConfig
except Exception as e:
    traceback.print_exc()
    raise

__all__ = ['Optimizer']
__all__.extend(
    filter(lambda x: x not in ['OptimizerType', 'BaseSGDOptimizer'],
           optimizer_types.__all__))


def __extends__(dict1, dict2):
    for key in dict2:
        assert key not in dict1
        dict1[key] = dict2[key]
    return dict1


class OptimizerConfig(object):
    def __init__(self, **kwargs):
        trainer_conf = TrainerConfig()
        self.conf_proto = trainer_conf.opt_config
        self.settings = dict(
            batch_size=None,
            mini_batch_size=None,
            algorithm='async_sgd',
            async_lagged_grad_discard_ratio=1.5,
            learning_method='momentum',
            num_batches_per_send_parameter=None,
            num_batches_per_get_parameter=None,
            center_parameter_update_method=None,
            learning_rate=1.,
            learning_rate_decay_a=0.,
            learning_rate_decay_b=0.,
            learning_rate_schedule='poly',
            learning_rate_args='',
            l1weight=0.1,
            l2weight=0.,
            l2weight_zero_iter=0,
            c1=0.0001,
            backoff=0.5,
            owlqn_steps=10,
            max_backoff=5,
            average_window=0,
            do_average_in_cpu=False,
            max_average_window=None,
            ada_epsilon=1e-6,
            ada_rou=0.95,
            delta_add_rate=1.0,
            shrink_parameter_value=0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8, )

        if 'batch_size' in kwargs:
            del kwargs['batch_size']  # not important for python library.

        self.extend_setting(batch_size=1, **kwargs)

    def update_setting(self, **args):
        for k, v in args.iteritems():
            if k in self.settings:
                self.settings[k] = v
            else:
                logger.fatal('Unkown setting: %s' % k)

        for k, v in self.settings.iteritems():
            if v is None:
                continue
            self.conf_proto.__setattr__(k, v)

    @default_decorators.wrap_param_default(
        ['learning_method'],
        default_factory=lambda _: optimizer_types.MomentumOptimizer())
    @default_decorators.wrap_param_default(
        ['regularization'],
        default_factory=lambda _: optimizer_types.BaseRegularization())
    def extend_setting(self,
                       batch_size,
                       learning_rate=1e-3,
                       learning_rate_decay_a=0.,
                       learning_rate_decay_b=0.,
                       learning_rate_schedule='poly',
                       learning_rate_args='',
                       learning_method=None,
                       regularization=None,
                       is_async=False,
                       model_average=None,
                       gradient_clipping_threshold=None):
        """
        Set the optimization method, learning rate, batch size, and other training
        settings. The currently supported algorithms are SGD and Async-SGD.

        ..  warning::

            Note that the 'batch_size' in PaddlePaddle is not equal to global
            training batch size. It represents the single training process's batch
            size. If you use N processes to train one model, for example use three
            GPU machines, the global batch size is N*'batch_size'.

        :param batch_size: batch size for one training process.
        :type batch_size: int
        :param learning_rate: learning rate for SGD
        :type learning_rate: float
        :param learning_method: The extension optimization algorithms of gradient
                                descent, such as momentum, adagrad, rmsprop, etc.
                                Note that it should be instance with base type
                                BaseSGDOptimizer.
        :type learning_method: BaseSGDOptimizer
        :param regularization: The regularization method.
        :type regularization: BaseRegularization
        :param is_async: Is Async-SGD or not. Default value is False.
        :type is_async: bool
        :param model_average: Model Average Settings.
        :type model_average: ModelAverage
        :param gradient_clipping_threshold: gradient clipping threshold. If gradient
                                            value larger than some value, will be
                                            clipped.
        :type gradient_clipping_threshold: float
        """
        if isinstance(regularization, optimizer_types.BaseRegularization):
            regularization = [regularization]

        assert isinstance(learning_method, optimizer_types.OptimizerType)
        if isinstance(learning_method, optimizer_types.BaseSGDOptimizer):
            algorithm = 'async_sgd' if is_async else 'sgd'
        else:
            algorithm = 'owlqn'

        args = [
            'batch_size', 'learning_rate', 'learning_rate_decay_a',
            'learning_rate_decay_b', 'learning_rate_schedule',
            'learning_rate_args'
        ]
        kwargs = dict()
        kwargs['algorithm'] = algorithm
        for arg in args:
            kwargs[arg] = locals()[arg]

        kwargs = __extends__(kwargs, learning_method.to_setting_kwargs())
        learning_method.extra_settings()

        for regular in regularization:
            assert isinstance(regular, optimizer_types.BaseRegularization)
            regular.algorithm = algorithm
            regular.learning_method = kwargs['learning_method']
            kwargs = __extends__(kwargs, regular.to_setting_kwargs())
            regular.extra_settings()

        if gradient_clipping_threshold is not None:
            gradient_clipping_threshold = optimizer_types.GradientClippingThreshold(
                threshold=gradient_clipping_threshold)

        for each in [model_average, gradient_clipping_threshold]:
            if each is not None:
                assert isinstance(each, optimizer_types.OptimizerType)
                each.algorithm = algorithm
                each.learning_method = kwargs['learning_method']
                kwargs = __extends__(kwargs, each.to_setting_kwargs())
                each.extra_settings()

        self.update_setting(**kwargs)

    def get_conf_proto(self):
        return self.conf_proto


class Optimizer(object):
    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            del kwargs['batch_size']  # not important for python library.

        self.__opt_conf_proto__ = OptimizerConfig(**kwargs).get_conf_proto()
        self.__opt_conf__ = api.OptimizationConfig.createFromProto(
            self.__opt_conf_proto__)

    def enable_types(self):
        tmp = api.ParameterOptimizer.create(self.__opt_conf__)
        assert isinstance(tmp, api.ParameterOptimizer)
        return tmp.getParameterTypes()

    def create_local_updater(self):
        return api.ParameterUpdater.createLocalUpdater(self.__opt_conf__)


if __name__ == '__main__':
    api.initPaddle('--use_gpu=false')
    opt = Optimizer(
        learning_rate=1e-4, learning_method=optimizer_types.AdamOptimizer())
    print opt.enable_types()

    optimizer = paddle.v2.optimizer.Optimizer(
        learning_method=paddle.v2.optimizer_types.AdamOptimizer(),
        learning_rate=1e-4,
        model_average=paddle.v2.optimizer_types.ModelAverage(
            average_window=0.5),
        regularization=paddle.v2.optimizer_types.L2Regularization(rate=0.5))

    print optimizer.enable_types()
