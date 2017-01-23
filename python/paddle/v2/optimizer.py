import py_paddle.swig_paddle as swig_api
import paddle.trainer_config_helpers.optimizers as v1_optimizers
import paddle.trainer_config_helpers.config_parser_utils as config_parser_utils
import paddle.v2

__all__ = ['Adam', 'Adamax']


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


if __name__ == '__main__':
    swig_api.initPaddle('--use_gpu=false')
    opt = paddle.v2.optimizer.Adam()
    print opt.enable_types()
