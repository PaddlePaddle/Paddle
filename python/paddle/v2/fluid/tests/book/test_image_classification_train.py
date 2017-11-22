import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
import paddle.v2.fluid.evaluator as evaluator
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.initializer import XavierInitializer
from paddle.v2.fluid.optimizer import AdamOptimizer


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
        tmp = layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=False)
        return layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride, program, init_program):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None, program,
                                 init_program)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None)
        short = shortcut(input, ch_in, ch_out, stride)
        return layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = layers.fc(input=drop,
                    size=512,
                    act=None,
                    param_attr={"initializer": XavierInitializer()})
    reshape1 = layers.reshape(x=fc1, shape=list(fc1.shape + (1, 1)))
    bn = layers.batch_norm(input=reshape1, act='relu')
    drop2 = layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = layers.fc(input=drop2,
                    size=512,
                    act=None,
                    param_attr={"initializer": XavierInitializer()})
    return fc2


classdim = 10
data_shape = [3, 32, 32]

images = layers.data(name='pixel', shape=data_shape, data_type='float32')
label = layers.data(name='label', shape=[1], data_type='int64')

# Add neural network config
# option 1. resnet
# net = resnet_cifar10(images, 32)
# option 2. vgg
net = vgg16_bn_drop(images)

# print(program)

predict = layers.fc(input=net, size=classdim, act='softmax')
cost = layers.cross_entropy(input=predict, label=label)
avg_cost = layers.mean(x=cost)

# optimizer = SGDOptimizer(learning_rate=0.001)
optimizer = AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

accuracy, acc_out = evaluator.accuracy(input=predict, label=label)

BATCH_SIZE = 128
PASS_NUM = 1

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(), buf_size=128 * 10),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(framework.default_startup_program())

for pass_id in range(PASS_NUM):
    batch_id = 0
    accuracy.reset(exe)
    for data in train_reader():
        img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        batch_size = 1
        for i in y_data.shape:
            batch_size = batch_size * i
        y_data = y_data.reshape([batch_size, 1])

        tensor_img = core.LoDTensor()
        tensor_y = core.LoDTensor()
        tensor_img.set(img_data, place)
        tensor_y.set(y_data, place)

        outs = exe.run(framework.default_main_program(),
                       feed={"pixel": tensor_img,
                             "label": tensor_y},
                       fetch_list=[avg_cost, acc_out])

        loss = np.array(outs[0])
        acc = np.array(outs[1])
        pass_acc = accuracy.eval(exe)
        print("pass_id:" + str(pass_id) + " batch_id:" + str(batch_id) +
              " loss:" + str(loss) + " acc:" + str(acc) + " pass_acc:" + str(
                  pass_acc))
        batch_id = batch_id + 1

        if batch_id > 1:
            # this model is slow, so if we can train two mini batch, we think it works properly.
            exit(0)
exit(1)
