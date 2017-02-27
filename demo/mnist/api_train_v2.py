import numpy
import paddle.v2 as paddle

import mnist_util


def train_reader():
    train_file = './data/raw_data/train'
    generator = mnist_util.read_from_mnist(train_file)
    for item in generator:
        yield item


def mlp(images):
    fc1 = paddle.layer.fc(input=images, size=200, act=paddle.activation.Relu())
    fc2 = paddle.layer.fc(input=fc1, size=200, act=paddle.activation.Relu())
    return fc2


def cnn(images):
    conv0 = paddle.layer.img_conv(
        input=images,
        filter_size=5,
        stride=1,
        num_channels=1,
        num_filters=20,
        bias_attr=True,
        act=paddle.activation.Relu())
    pool0 = paddle.layer.img_pool(
        input=conv0, pool_size=2, stride=2, pool_type=paddle.pooling.Max())

    conv1 = paddle.layer.img_conv(
        input=pool0,
        filter_size=5,
        stride=1,
        num_filters=50,
        bias_attr=True,
        act=paddle.activation.Relu())
    pool1 = paddle.layer.img_pool(
        input=conv1, pool_size=2, stride=2, pool_type=paddle.pooling.Max())
    fc1 = paddle.layer.fc(input=pool1, size=128)

    return fc1


def main():
    paddle.init(use_gpu=True, trainer_count=1)
    use_cnn = False

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    out = cnn(images) if use_cnn else mlp(images)
    inference = paddle.layer.fc(input=out,
                                size=10,
                                act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=inference, label=label)

    parameters = paddle.parameters.create(cost)

    #optimizer = paddle.optimizer.Adam(learning_rate=0.01)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0, momentum=0.9)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=optimizer)

    trainer.train(train_data_reader=train_reader,
                  topology=cost,
                  parameters=parameters,
                  event_handler=event_handler,
                  batch_size=128,  # batch size should be refactor in Data reader
                  data_types=[  # data_types will be removed, It should be in
                      # network topology
                      ('pixel', images.type),
                      ('label', label.type)],
                  reader_dict={'pixel':0, 'label':1},
                  num_passes=20,
                  )


if __name__ == '__main__':
    main()
