import paddle.v2 as paddle

import mnist_util


def train_reader():
    train_file = './data/raw_data/train'
    generator = mnist_util.read_from_mnist(train_file)
    for item in generator:
        yield item


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    hidden1 = paddle.layer.fc(input=images, size=200)
    hidden2 = paddle.layer.fc(input=hidden1, size=200)
    inference = paddle.layer.fc(input=hidden2,
                                size=10,
                                act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=inference, label=label)

    parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.01)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=adam_optimizer)

    trainer.train(
        train_data_reader=train_reader,
        topology=[cost],
        parameters=parameters,
        event_handler=event_handler,
        batch_size=32)  # batch size should be refactor in Data reader


if __name__ == '__main__':
    main()
