"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""

import paddle.trainer.PyDataProvider2 as dp
from paddle.trainer_config_helpers import *

import mnist_provider
from py_paddle.trainer import *


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


def main():
    mnist = mnist_network()
    runner = RunnerBuilder(
        network=mnist, device_count=2).with_std_local_trainer(
            method=mnist_provider.process,
            file_list=['./data/raw_data/train']).with_std_tester(
                method=mnist_provider.process,
                file_list=['./data/raw_data/t10k']).build()
    with runner:
        for _ in xrange(2):
            runner.run_one_pass()


if __name__ == '__main__':
    main()
