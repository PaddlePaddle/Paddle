"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""

import paddle.v2 as paddle

from mnist_util import read_from_mnist


def main():
    paddle.raw.initPaddle("-use_gpu=false",
                          "-trainer_count=4")  # use 4 cpu cores

    optimizer = paddle.optimizer.Optimizer(
        learning_method=paddle.optimizer.AdamOptimizer(),
        learning_rate=1e-4,
        model_average=paddle.optimizer.ModelAverage(average_window=0.5),
        regularization=paddle.optimizer.L2Regularization(rate=0.5))

    # define network
    imgs = paddle.layers.data_layer(name='pixel', size=784)
    hidden1 = paddle.layers.fc_layer(input=imgs, size=200)
    hidden2 = paddle.layers.fc_layer(input=hidden1, size=200)
    inference = paddle.layers.fc_layer(
        input=hidden2, size=10, act=paddle.config.SoftmaxActivation())
    cost = paddle.layers.classification_cost(
        input=inference, label=paddle.layers.data_layer(
            name='label', size=10))

    model = paddle.model.Model(layers=[cost], optimizer=optimizer)

    model.rand_parameter()

    batch_evaluator = model.make_evaluator()
    test_evaluator = model.make_evaluator()

    train_data = paddle.data.create_data_pool(
        file_reader=read_from_mnist,
        file_list=['./data/raw_data/train'],
        model=model,
        batch_size=128,
        shuffle=True)
    test_data = paddle.data.create_data_pool(
        file_reader=read_from_mnist,
        file_list=['./data/raw_data/test'],
        model=model,
        batch_size=128,
        shuffle=False)

    # Training process.
    model.start()

    for pass_id in xrange(2):
        model.start_pass()

        for batch_id, data_batch in enumerate(train_data):
            model.start_batch()
            model.train(data_batch)
            batch_evaluator.start()
            model.evaluate(batch_evaluator)
            batch_evaluator.finish()
            print "Pass=%d, batch=%d" % (pass_id, batch_id), batch_evaluator
            model.finish_batch()

        test_evaluator.start()
        for _, data_batch in enumerate(test_data):
            model.test(data_batch)
        print "TEST Pass=%d" % pass_id, test_evaluator
        test_evaluator.finish()

        model.finish_pass()

    model.finish()


if __name__ == '__main__':
    main()
