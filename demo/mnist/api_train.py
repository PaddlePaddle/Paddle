"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""

import mnist_provider
import paddle.trainer.PyDataProvider2 as dp
import py_paddle.swig_paddle as api
from mnist_util import read_from_mnist
from paddle.trainer_config_helpers import *
from trainer import *


@network(
    inputs={
        'pixel': dp.dense_vector(784),
        'label': dp.integer_value(10),
    },
    learning_rate=1e-4,
    learning_method=AdamOptimizer(),
    batch_size=1000,
    model_average=ModelAverage(average_window=0.5),
    regularization=L2Regularization(rate=0.5))
def mnist_network(pixel, label):
    hidden1 = fc_layer(input=pixel, size=200)
    hidden2 = fc_layer(input=hidden1, size=200)
    inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    cost = classification_cost(input=inference, label=label)
    return cost


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


class MonolithicChainItem(RunnerChainItem):
    def finalize(self, context, next_callback):
        context.gradient_machine.finish()

    def initialize(self, context, next_callback):
        test_file = './data/raw_data/t10k'

        context.test_file = test_file

        next_callback(context)

    def on_batch_begin(self, context, next_callback):
        batch_evaluator = context.gradient_machine.makeEvaluator()
        # outArgs is Neural Network forward result. Here is not useful, just passed
        # to gradient_machine.forward
        outArgs = api.Arguments.createArguments(0)

        # Start BatchEvaluator.
        # batch_evaluator can be used between start/finish.
        batch_evaluator.start()

        # forwardBackward is a shortcut for forward and backward.
        # It is sometimes faster than invoke forward/backward separately,
        # because in GradientMachine, it may be async.
        context.gradient_machine.forwardBackward(context.in_args, outArgs,
                                                 api.PASS_TRAIN)

        for each_param in context.gradient_machine.getParameters():
            context.updater.update(each_param)

        # Get cost. We use numpy to calculate total cost for this batch.
        cost_vec = outArgs.getSlotValue(0)
        cost_vec = cost_vec.copyToNumpyMat()
        cost = cost_vec.sum() / context.current_batch_size
        context.current_cost = cost

        # Make evaluator works.
        context.gradient_machine.eval(batch_evaluator)

        # Print logs.
        print 'batch with cost=', cost, batch_evaluator

        batch_evaluator.finish()
        context.cost = cost
        return False

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

    def __init__(self):
        RunnerChainItem.__init__(self)


def main():
    mnist = mnist_network()

    runner = Runner()
    runner.add_chain_item(DeviceChainItem(use_gpu=False, device_count=4))

    runner.add_chain_item(CreateGradientMachine(network=mnist))
    runner.add_chain_item(RandomInitializeParams())
    runner.add_chain_item(
        BasicTrainerDataProvider(
            network=mnist,
            method=mnist_provider.process,
            file_list=['./data/raw_data/train'],
            batch_size=256))
    runner.add_chain_item(BasicLocalParameterUpdater(network=mnist))

    runner.add_chain_item(MonolithicChainItem())
    with runner.use():
        for _ in xrange(2):
            runner.run_one_pass()


if __name__ == '__main__':
    main()
