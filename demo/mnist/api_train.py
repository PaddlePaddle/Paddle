"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""
import random

import paddle.v2 as paddle

from mnist_util import read_from_mnist


def network_config():
    imgs = paddle.config.data_layer(name='pixel', size=784)
    hidden1 = paddle.config.fc_layer(input=imgs, size=200)
    hidden2 = paddle.config.fc_layer(input=hidden1, size=200)
    inference = paddle.config.fc_layer(
        input=hidden2, size=10, act=paddle.config.SoftmaxActivation())
    cost = paddle.config.classification_cost(
        input=inference, label=paddle.config.data_layer(
            name='label', size=10))
    paddle.config.outputs(cost)


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
    paddle.raw.initPaddle("-use_gpu=false",
                          "-trainer_count=4")  # use 4 cpu cores

    # get enable_types for each optimizer.
    # enable_types = [value, gradient, momentum, etc]
    # For each optimizer(SGD, Adam), GradientMachine should enable different
    # buffers.
    optimizer = paddle.optimizer.Optimizer(
        learning_method=paddle.optimizer.AdamOptimizer(),
        learning_rate=1e-4,
        model_average=paddle.optimizer.ModelAverage(average_window=0.5),
        regularization=paddle.optimizer.L2Regularization(rate=0.5))

    # Create Simple Gradient Machine.
    model_config = paddle.config.parse_network(network_config)
    m = paddle.raw.GradientMachine.createFromConfigProto(
        model_config, paddle.raw.CREATE_MODE_NORMAL, optimizer.enable_types())

    # This type check is not useful. Only enable type hint in IDE.
    # Such as PyCharm
    assert isinstance(m, paddle.raw.GradientMachine)

    # Initialize Parameter by numpy.
    m.randParameters()

    # Create Local Updater. Local means not run in cluster.
    # For a cluster training, here we can change to createRemoteUpdater
    # in future.
    updater = optimizer.create_local_updater()
    assert isinstance(updater, paddle.raw.ParameterUpdater)

    # Initialize ParameterUpdater.
    updater.init(m)

    # DataProvider Converter is a utility convert Python Object to Paddle C++
    # Input. The input format is as same as Paddle's DataProvider.
    converter = paddle.data.DataProviderConverter(input_types=[
        paddle.data.dense_vector(784), paddle.data.integer_value(10)
    ])

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
    outArgs = paddle.raw.Arguments.createArguments(0)

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
            m.forward(converter(data_batch), outArgs, paddle.raw.PASS_TEST)
            m.eval(test_evaluator)

        # print error rate for test data set
        print 'Pass', pass_id, ' test evaluator: ', test_evaluator
        test_evaluator.finish()
        updater.restore()

        updater.catchUpWith()
        params = m.getParameters()
        for each_param in params:
            assert isinstance(each_param, paddle.raw.Parameter)
            value = each_param.getBuf(paddle.raw.PARAMETER_VALUE)
            value = value.copyToNumpyArray()

            # Here, we could save parameter to every where you want
            print each_param.getName(), value

        updater.finishPass()

    m.finish()


if __name__ == '__main__':
    main()
