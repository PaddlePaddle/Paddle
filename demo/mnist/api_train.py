"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""
import py_paddle.swig_paddle as api
from py_paddle import DataProviderConverter
import paddle.trainer.PyDataProvider2 as dp
import numpy as np
import random
from mnist_util import read_from_mnist
from paddle.trainer_config_helpers import *

from trainer import *


def optimizer_config():
    settings(
        learning_rate=1e-4,
        learning_method=AdamOptimizer(),
        batch_size=1000,
        model_average=ModelAverage(average_window=0.5),
        regularization=L2Regularization(rate=0.5))


def network_config():
    imgs = data_layer(name='pixel', size=784)
    hidden1 = fc_layer(input=imgs, size=200)
    hidden2 = fc_layer(input=hidden1, size=200)
    inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    cost = classification_cost(
        input=inference, label=data_layer(
            name='label', size=10))
    outputs(cost)


def init_parameter(network):
    assert isinstance(network, api.GradientMachine)
    for each_param in network.getParameters():
        assert isinstance(each_param, api.Parameter)
        array_size = len(each_param)
        array = np.random.uniform(-1.0, 1.0, array_size).astype('float32')
        each_param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(array)


def generator_to_batch(generator, batch_size):
    ret_val = list()
    for each_item in generator:
        ret_val.append(each_item)
        if len(ret_val) == batch_size:
            yield ret_val
            ret_val = list()
    if len(ret_val) != 0:
        yield ret_val


class BatchPool(object):
    def __init__(self, generator, batch_size):
        self.data = list(generator)
        self.batch_size = batch_size

    def __call__(self):
        random.shuffle(self.data)
        for offset in xrange(0, len(self.data), self.batch_size):
            limit = min(offset + self.batch_size, len(self.data))
            yield self.data[offset:limit]


def input_order_converter(generator):
    for each_item in generator:
        yield each_item['pixel'], each_item['label']


class MonolithicChainItem(RunnerChainItem):
    def finalize(self, context, next_callback):
        context.gradient_machine.finish()

    def initialize(self, context, next_callback):
        api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores

        # get enable_types for each optimizer.
        # enable_types = [value, gradient, momentum, etc]
        # For each optimizer(SGD, Adam), GradientMachine should enable different
        # buffers.
        opt_config_proto = parse_optimizer_config(optimizer_config)
        opt_config = api.OptimizationConfig.createFromProto(opt_config_proto)
        _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
        enable_types = _temp_optimizer_.getParameterTypes()

        # Create Simple Gradient Machine.
        model_config = parse_network_config(network_config)
        context.gradient_machine = api.GradientMachine.createFromConfigProto(
            model_config, api.CREATE_MODE_NORMAL, enable_types)

        # This type check is not useful. Only enable type hint in IDE.
        # Such as PyCharm
        assert isinstance(context.gradient_machine, api.GradientMachine)

        # Initialize Parameter by numpy.
        init_parameter(network=context.gradient_machine)

        # Create Local Updater. Local means not run in cluster.
        # For a cluster training, here we can change to createRemoteUpdater
        # in future.
        context.updater = api.ParameterUpdater.createLocalUpdater(opt_config)
        assert isinstance(context.updater, api.ParameterUpdater)
        context.updater.init(context.gradient_machine)

        # DataProvider Converter is a utility convert Python Object to Paddle C++
        # Input. The input format is as same as Paddle's DataProvider.
        context.data_converter = DataProviderConverter(
            input_types=[dp.dense_vector(784), dp.integer_value(10)])

        train_file = './data/raw_data/train'
        test_file = './data/raw_data/t10k'

        context.gradient_machine.start()

        # Get Train Data.
        # TrainData will stored in a data pool. Currently implementation is not care
        # about memory, speed. Just a very naive implementation.
        train_data_generator = input_order_converter(
            read_from_mnist(train_file))
        train_data = BatchPool(train_data_generator, 512)
        context.train_data_callback = train_data
        context.test_file = test_file

        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        batch_evaluator = context.gradient_machine.makeEvaluator()
        # outArgs is Neural Network forward result. Here is not useful, just passed
        # to gradient_machine.forward
        outArgs = api.Arguments.createArguments(0)

        try:
            data_batch = next(context.train_data)
        except StopIteration:
            return True

        # data_batch is input images.
        # here, for online learning, we could get data_batch from network.

        # Start update one batch.
        pass_type = context.updater.startBatch(len(data_batch))

        # Start BatchEvaluator.
        # batch_evaluator can be used between start/finish.
        batch_evaluator.start()

        # forwardBackward is a shortcut for forward and backward.
        # It is sometimes faster than invoke forward/backward separately,
        # because in GradientMachine, it may be async.
        context.gradient_machine.forwardBackward(
            context.data_converter(data_batch), outArgs, pass_type)

        for each_param in context.gradient_machine.getParameters():
            context.updater.update(each_param)

        # Get cost. We use numpy to calculate total cost for this batch.
        cost_vec = outArgs.getSlotValue(0)
        cost_vec = cost_vec.copyToNumpyMat()
        cost = cost_vec.sum() / len(data_batch)

        # Make evaluator works.
        context.gradient_machine.eval(batch_evaluator)

        # Print logs.
        print 'batch with cost=', cost, batch_evaluator

        batch_evaluator.finish()
        context.cost = cost
        return False

    def on_pass_begin(self, context, next_callback):
        context.updater.startPass()
        context.train_data = context.train_data_callback()

    def on_pass_end(self, context, next_callback):
        # testing stage. use test data set to test current network.
        outArgs = api.Arguments.createArguments(0)
        context.updater.apply()
        test_evaluator = context.gradient_machine.makeEvaluator()
        test_evaluator.start()
        test_data_generator = input_order_converter(
            read_from_mnist(context.test_file))
        for data_batch in generator_to_batch(test_data_generator, 512):
            # in testing stage, only forward is needed.
            context.gradient_machine.forward(
                context.data_converter(data_batch), outArgs, api.PASS_TEST)
            context.gradient_machine.eval(test_evaluator)

        # print error rate for test data set
        print 'Test evaluator: ', test_evaluator
        test_evaluator.finish()
        context.updater.restore()

        context.updater.catchUpWith()
        params = context.gradient_machine.getParameters()
        for each_param in params:
            assert isinstance(each_param, api.Parameter)
            value = each_param.getBuf(api.PARAMETER_VALUE)
            value = value.copyToNumpyArray()

            # Here, we could save parameter to every where you want
            print each_param.getName(), value

        context.updater.finishPass()

    def on_batch_end(self, context, next_callback):
        # Finish batch.
        #  * will clear gradient.
        #  * ensure all values should be updated.
        context.updater.finishBatch(context.cost)
        return False

    def __init__(self):
        RunnerChainItem.__init__(self)


def main():
    runner = Runner()
    runner.add_chain_item(MonolithicChainItem())
    with runner.use():
        for _ in xrange(2):
            runner.run_one_pass()


if __name__ == '__main__':
    main()
