from paddle.trainer_config_helpers import *
import py_paddle.swig_paddle as api


class Optimizer(object):
    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            del kwargs['batch_size']  # not important for python library.

        def __impl__():
            settings(batch_size=1, **kwargs)

        self.__opt_conf_proto__ = parse_optimizer_config(__impl__)
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
    opt = Optimizer(learning_rate=1e-4, learning_method=AdamOptimizer())
    print opt.enable_types()
