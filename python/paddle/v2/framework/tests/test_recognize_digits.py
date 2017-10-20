import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np

init_program = Program()
program = Program()

images = layers.data(
    name='pixel',
    shape=[1, 28, 28],
    data_type='float32',
    program=program,
    init_program=init_program)
label = layers.data(
    name='label',
    shape=[1],
    data_type='int32',
    program=program,
    init_program=init_program)
conv_pool_1 = nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=2,
    pool_size=2,
    pool_stride=2,
    act="relu",
    program=program,
    init_program=init_program)
conv_pool_2 = nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=4,
    pool_size=2,
    pool_stride=2,
    act="relu",
    program=program,
    init_program=init_program)

predict = layers.fc(input=conv_pool_2,
                    size=10,
                    act="softmax",
                    program=program,
                    init_program=init_program)
cost = layers.cross_entropy(
    input=predict, label=label, program=program, init_program=init_program)
avg_cost = layers.mean(x=cost, program=program)

sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
opts = sgd_optimizer.minimize(avg_cost)

BATCH_SIZE = 20
PASS_NUM = 100

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(init_program, feed={}, fetch_list=[])

for pass_id in range(PASS_NUM):
    for data in train_reader():
        print data
