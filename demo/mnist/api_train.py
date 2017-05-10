"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""
import random

import numpy as np
import paddle.v2 as paddle_v2
import py_paddle.swig_paddle as api
from paddle.trainer_config_helpers import *
from py_paddle import DataProviderConverter

from mnist_util import read_from_mnist


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


def main():
    api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores

    optimizer = paddle_v2.optimizer.Adam(
        learning_rate=1e-4,
        batch_size=1000,
        model_average=ModelAverage(average_window=0.5),
        regularization=L2Regularization(rate=0.5))

    # Create Local Updater. Local means not run in cluster.
    # For a cluster training, here we can change to createRemoteUpdater
    # in future.
    updater = optimizer.create_local_updater()
    assert isinstance(updater, api.ParameterUpdater)

    # define network
    images = paddle_v2.layer.data(
        name='pixel', type=paddle_v2.data_type.dense_vector(784))
    label = paddle_v2.layer.data(
        name='label', type=paddle_v2.data_type.integer_value(10))
    hidden1 = paddle_v2.layer.fc(input=images, size=200)
    hidden2 = paddle_v2.layer.fc(input=hidden1, size=200)
    inference = paddle_v2.layer.fc(input=hidden2,
                                   size=10,
                                   act=paddle_v2.activation.Softmax())
    cost = paddle_v2.layer.classification_cost(input=inference, label=label)

    # Create Simple Gradient Machine.
    model_config = paddle_v2.layer.parse_network(cost)
    m = api.GradientMachine.createFromConfigProto(model_config,
                                                  api.CREATE_MODE_NORMAL,
                                                  optimizer.enable_types())

    # This type check is not useful. Only enable type hint in IDE.
    # Such as PyCharm
    assert isinstance(m, api.GradientMachine)

    # Initialize Parameter by numpy.
    init_parameter(network=m)

    # Initialize ParameterUpdater.
    updater.init(m)

    # DataProvider Converter is a utility convert Python Object to Paddle C++
    # Input. The input format is as same as Paddle's DataProvider.
    converter = DataProviderConverter(input_types=[images.type, label.type])

    train_file = './data/raw_data/train'
    test_file = './data/raw_data/t10k'

    # start gradient machine.
    # the gradient machine must be started before invoke forward/backward.
    # not just for training, but also for inference.
    m.start()

    # evaluator can print error rate, etc. It is a C++ class.
    batch_evaluator = m.makeEvaluator()
    test_evaluator = m.makeEvaluator()

    # Get Train Data.
    # TrainData will stored in a data pool. Currently implementation is not care
    # about memory, speed. Just a very naive implementation.
    train_data_generator = input_order_converter(read_from_mnist(train_file))
    train_data = BatchPool(train_data_generator, 512)

    # outArgs is Neural Network forward result. Here is not useful, just passed
    # to gradient_machine.forward
    outArgs = api.Arguments.createArguments(0)

    for pass_id in xrange(2):  # we train 2 passes.
        updater.startPass()

        for batch_id, data_batch in enumerate(train_data()):
            # data_batch is input images.
            # here, for online learning, we could get data_batch from network.

            # Start update one batch.
            pass_type = updater.startBatch(len(data_batch))

            # Start BatchEvaluator.
            # batch_evaluator can be used between start/finish.
            batch_evaluator.start()

            # forwardBackward is a shortcut for forward and backward.
            # It is sometimes faster than invoke forward/backward separately,
            # because in GradientMachine, it may be async.
            m.forwardBackward(converter(data_batch), outArgs, pass_type)

            for each_param in m.getParameters():
                updater.update(each_param)

            # Get cost. We use numpy to calculate total cost for this batch.
            cost_vec = outArgs.getSlotValue(0)
            cost_vec = cost_vec.copyToNumpyMat()
            cost = cost_vec.sum() / len(data_batch)

            # Make evaluator works.
            m.eval(batch_evaluator)

            # Print logs.
            print 'Pass id', pass_id, 'Batch id', batch_id, 'with cost=', \
                cost, batch_evaluator

            batch_evaluator.finish()
            # Finish batch.
            #  * will clear gradient.
            #  * ensure all values should be updated.
            updater.finishBatch(cost)

        # testing stage. use test data set to test current network.
        updater.apply()
        test_evaluator.start()
        test_data_generator = input_order_converter(read_from_mnist(test_file))
        for data_batch in generator_to_batch(test_data_generator, 512):
            # in testing stage, only forward is needed.
            m.forward(converter(data_batch), outArgs, api.PASS_TEST)
            m.eval(test_evaluator)

        # print error rate for test data set
        print 'Pass', pass_id, ' test evaluator: ', test_evaluator
        test_evaluator.finish()
        updater.restore()

        updater.catchUpWith()
        params = m.getParameters()
        for each_param in params:
            assert isinstance(each_param, api.Parameter)
            value = each_param.getBuf(api.PARAMETER_VALUE)
            value = value.copyToNumpyArray()

            # Here, we could save parameter to every where you want
            print each_param.getName(), value

        updater.finishPass()

    m.finish()


if __name__ == '__main__':
    main()
