import py_paddle.swig_paddle as swig_api

import paddle.trainer_config_helpers.config_parser_utils as config_parser_utils
import paddle.trainer_config_helpers.optimizers as v1_optimizers
"""
Optimizers(update equation) for SGD method.

TODO(yuyang18): Complete comments.
"""

__all__ = [
    'Momentum', 'Adam', 'Adamax', 'AdaGrad', 'DecayedAdaGrad', 'AdaDelta',
    'RMSProp', 'ModelAverage', 'L2Regularization'
]


class Optimizer(object):
    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            del kwargs['batch_size']  # not important for python library.

        def __impl__():
            v1_optimizers.settings(batch_size=1, **kwargs)

        self.__opt_conf_proto__ = config_parser_utils.parse_optimizer_config(
            __impl__)
        self.__opt_conf__ = swig_api.OptimizationConfig.createFromProto(
            self.__opt_conf_proto__)

    def enable_types(self):
        """
        get enable_types for each optimizer.
        enable_types = [value, gradient, momentum, etc]
        For each optimizer(SGD, Adam), GradientMachine should enable different
        buffers.
        """
        tmp = swig_api.ParameterOptimizer.create(self.__opt_conf__)
        assert isinstance(tmp, swig_api.ParameterOptimizer)
        return tmp.getParameterTypes()

    def create_local_updater(self):
        return swig_api.ParameterUpdater.createLocalUpdater(self.__opt_conf__)

    def create_remote_updater(self, pass_num):
        return swig_api.ParameterUpdater.createRemoteUpdater(self.__opt_conf__,
                                                             pass_num)


class Momentum(Optimizer):
    def __init__(self, momentum=None, sparse=False, **kwargs):
        learning_method = v1_optimizers.MomentumOptimizer(
            momentum=momentum, sparse=sparse)
        super(Momentum, self).__init__(
            learning_method=learning_method, **kwargs)


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        learning_method = v1_optimizers.AdamOptimizer(
            beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(Adam, self).__init__(learning_method=learning_method, **kwargs)


class Adamax(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        learning_method = v1_optimizers.AdamaxOptimizer(
            beta1=beta1, beta2=beta2)
        super(Adamax, self).__init__(learning_method=learning_method, **kwargs)


class AdaGrad(Optimizer):
    def __init__(self, **kwargs):
        learning_method = v1_optimizers.AdaGradOptimizer()
        super(AdaGrad, self).__init__(learning_method=learning_method, **kwargs)


class DecayedAdaGrad(Optimizer):
    def __init__(self, rho=0.95, epsilon=1e-06, **kwargs):
        learning_method = v1_optimizers.DecayedAdaGradOptimizer(
            rho=rho, epsilon=epsilon)
        super(DecayedAdaGrad, self).__init__(
            learning_method=learning_method, **kwargs)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, epsilon=1e-06, **kwargs):
        learning_method = v1_optimizers.AdaDeltaOptimizer(
            rho=rho, epsilon=epsilon)
        super(AdaDelta, self).__init__(
            learning_method=learning_method, **kwargs)


class RMSProp(Optimizer):
    def __init__(self, rho=0.95, epsilon=1e-6, **kwargs):
        learning_method = v1_optimizers.RMSPropOptimizer(
            rho=rho, epsilon=epsilon)
        super(RMSProp, self).__init__(learning_method=learning_method, **kwargs)


ModelAverage = v1_optimizers.ModelAverage
L2Regularization = v1_optimizers.L2Regularization

if __name__ == '__main__':
    swig_api.initPaddle('--use_gpu=false')
    for opt in [
            Momentum(), Adam(), Adamax(), AdaGrad(), DecayedAdaGrad(),
            AdaDelta(), RMSProp(), Adam(
                model_average=ModelAverage(average_window=0.5),
                regularization=L2Regularization(rate=0.5),
                gradient_clipping_threshold=25)
    ]:
        print opt, opt.enable_types()
