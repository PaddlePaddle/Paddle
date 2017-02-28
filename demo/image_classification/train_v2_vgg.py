import paddle.v2 as paddle


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)
    else:
        pass


def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.layer.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=pooling.Max())

    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(
        input=fc1,
        act=paddle.activation.Relu(),
        layer_attr=ExtraAttr(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
    return fc2


def main():
    datadim = 3 * 32 * 32
    classdim = 10

    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))
    # net = vgg_bn_drop(image)
    out = paddle.layer.fc(input=image,
                          size=classdim,
                          act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    parameters = paddle.parameters.create(cost)
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp',
        batch_size=128)

    trainer = paddle.trainer.SGD(update_equation=momentum_optimizer)
    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size=3072),
            batch_size=128),
        cost=cost,
        num_passes=1,
        parameters=parameters,
        event_handler=event_handler,
        reader_dict={'image': 0,
                     'label': 1}, )


if __name__ == '__main__':
    main()
