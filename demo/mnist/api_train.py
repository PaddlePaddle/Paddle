import py_paddle.swig_paddle as api
import paddle.trainer.config_parser
import numpy as np


def init_parameter(network):
    assert isinstance(network, api.GradientMachine)
    for each_param in network.getParameters():
        assert isinstance(each_param, api.Parameter)
        array = each_param.getBuf(api.PARAMETER_VALUE).toNumpyArrayInplace()
        assert isinstance(array, np.ndarray)
        for i in xrange(len(array)):
            array[i] = np.random.uniform(-1.0, 1.0)


def main():
    api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores
    config = paddle.trainer.config_parser.parse_config(
        'simple_mnist_network.py', '')

    opt_config = api.OptimizationConfig.createFromProto(config.opt_config)
    _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
    enable_types = _temp_optimizer_.getParameterTypes()

    m = api.GradientMachine.createFromConfigProto(
        config.model_config, api.CREATE_MODE_NORMAL, enable_types)
    assert isinstance(m, api.GradientMachine)
    init_parameter(network=m)

    updater = api.ParameterUpdater.createLocalUpdater(opt_config)
    assert isinstance(updater, api.ParameterUpdater)
    updater.init(m)
    updater.startPass()


if __name__ == '__main__':
    main()
