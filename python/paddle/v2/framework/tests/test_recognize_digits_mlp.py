import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np

init_program = Program()
program = Program()
image = layers.data(
    name='x',
    shape=[784],
    data_type='float32',
    program=program,
    init_program=init_program)

hidden1 = layers.fc(input=image,
                    size=128,
                    act='relu',
                    program=program,
                    init_program=init_program)
hidden2 = layers.fc(input=hidden1,
                    size=64,
                    act='relu',
                    program=program,
                    init_program=init_program)

predict = layers.fc(input=hidden2,
                    size=10,
                    act='softmax',
                    program=program,
                    init_program=init_program)

label = layers.data(
    name='y',
    shape=[1],
    data_type='int64',
    program=program,
    init_program=init_program)

cost = layers.cross_entropy(
    input=predict, label=label, program=program, init_program=init_program)
avg_cost = layers.mean(x=cost, program=program, init_program=init_program)

sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
opts = sgd_optimizer.minimize(avg_cost)

BATCH_SIZE = 128

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(init_program, feed={}, fetch_list=[])

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    for data in train_reader():
        x_data = np.array(map(lambda x: x[0], data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = np.expand_dims(y_data, axis=1)

        tensor_x = core.LoDTensor()
        tensor_x.set(x_data, place)

        tensor_y = core.LoDTensor()
        tensor_y.set(y_data, place)

        outs = exe.run(program,
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost])
        out = np.array(outs[0])
        if out[0] < 5.0:
            exit(0)  # if avg cost less than 5.0, we think our code is good.
exit(1)
