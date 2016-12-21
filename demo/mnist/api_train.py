import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter
import paddle.trainer.PyDataProvider2 as dp
import paddle.trainer.config_parser
import numpy as np
from mnist_util import read_from_mnist


def init_parameter(network):
    assert isinstance(network, api.GradientMachine)
    for each_param in network.getParameters():
        assert isinstance(each_param, api.Parameter)
        array = each_param.getBuf(api.PARAMETER_VALUE).toNumpyArrayInplace()
        assert isinstance(array, np.ndarray)
        for i in xrange(len(array)):
            array[i] = np.random.uniform(-1.0, 1.0)


def generator_to_batch(generator, batch_size):
    ret_val = list()
    for each_item in generator:
        ret_val.append(each_item)
        if len(ret_val) == batch_size:
            yield ret_val
            ret_val = list()
    if len(ret_val) != 0:
        yield ret_val


def input_order_converter(generator):
    for each_item in generator:
        yield each_item['pixel'], each_item['label']


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

    converter = DataProviderConverter(
        input_types=[dp.dense_vector(784), dp.integer_value(10)])

    train_file = './data/raw_data/train'

    m.start()

    for _ in xrange(100):
        updater.startPass()
        outArgs = api.Arguments.createArguments(0)
        train_data_generator = input_order_converter(
            read_from_mnist(train_file))
        for batch_id, data_batch in enumerate(
                generator_to_batch(train_data_generator, 2048)):
            trainRole = updater.startBatch(len(data_batch))

            def updater_callback(param):
                updater.update(param)

            m.forwardBackward(
                converter(data_batch), outArgs, trainRole, updater_callback)

            cost_vec = outArgs.getSlotValue(0)
            cost_vec = cost_vec.copyToNumpyMat()
            cost = cost_vec.sum() / len(data_batch)
            print 'Batch id', batch_id, 'with cost=', cost
            updater.finishBatch(cost)

        updater.finishPass()

    m.finish()


if __name__ == '__main__':
    main()
