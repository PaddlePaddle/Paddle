import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np
import sys


# disable buffer
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


def conv_block(input,
               num_filter,
               groups,
               dropouts,
               program=None,
               init_program=None):
    return nets.img_conv_group(
        input=input,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[num_filter] * groups,
        conv_filter_size=3,
        conv_act='relu',
        conv_with_batchnorm=True,
        conv_batchnorm_drop_rate=dropouts,
        pool_type='max',
        program=program,
        init_program=init_program)


init_program = Program()
program = Program()

classdim = 10
data_shape = [3, 32, 32]

images = layers.data(
    name='pixel', shape=data_shape, data_type='float32', program=program)

label = layers.data(
    name='label',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)

conv1 = conv_block(images, 64, 2, [0.3, 0], program, init_program)
conv2 = conv_block(conv1, 128, 2, [0.4, 0], program, init_program)
conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0], program, init_program)
conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0], program, init_program)
conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0], program, init_program)
#
drop = layers.dropout(
    x=conv5, dropout_prob=0.5, program=program, init_program=init_program)
fc1 = layers.fc(input=drop,
                size=512,
                act=None,
                program=program,
                init_program=init_program)
reshape1 = layers.reshape(
    x=fc1,
    shape=list(fc1.shape + (1, 1)),
    program=program,
    init_program=init_program)
bn = layers.batch_norm(
    input=reshape1, act='relu', program=program, init_program=init_program)
drop2 = layers.dropout(
    x=bn, dropout_prob=0.5, program=program, init_program=init_program)
fc2 = layers.fc(input=drop2,
                size=512,
                act=None,
                program=program,
                init_program=init_program)
predict = layers.fc(input=fc2,
                    size=classdim,
                    act='softmax',
                    program=program,
                    init_program=init_program)

cost = layers.cross_entropy(
    input=predict, label=label, program=program, init_program=init_program)
avg_cost = layers.mean(x=cost, program=program, init_program=init_program)

sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
opts = sgd_optimizer.minimize(avg_cost)

BATCH_SIZE = 128
PASS_NUM = 1

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(), buf_size=128 * 10),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(init_program, feed={}, fetch_list=[])

for pass_id in range(PASS_NUM):
    count = 0
    for data in train_reader():
        img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([BATCH_SIZE, 1])

        tensor_img = core.LoDTensor()
        tensor_y = core.LoDTensor()
        tensor_img.set(img_data, place)
        tensor_y.set(y_data, place)

        outs = exe.run(program,
                       feed={"pixel": tensor_img,
                             "label": tensor_y},
                       fetch_list=[avg_cost])

        loss = np.array(outs[0])
        print(loss)

        if loss < 0.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
