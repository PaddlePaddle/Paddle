import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.regularizer import L2DecayRegularizer
from paddle.v2.framework.initializer import UniformInitializer

import numpy as np

BATCH_SIZE = 128
startup_program = Program()
main_program = Program()
image = layers.data(
    name='x',
    shape=[784],
    data_type='float32',
    main_program=main_program,
    startup_program=startup_program)

param_attr = {
    'name': None,
    'initializer': UniformInitializer(
        low=-1.0, high=1.0),
    'regularization': L2DecayRegularizer(0.0005 * BATCH_SIZE)
}

hidden1 = layers.fc(input=image,
                    size=128,
                    act='relu',
                    main_program=main_program,
                    startup_program=startup_program,
                    param_attr=param_attr)
hidden2 = layers.fc(input=hidden1,
                    size=64,
                    act='relu',
                    main_program=main_program,
                    startup_program=startup_program,
                    param_attr=param_attr)

predict = layers.fc(input=hidden2,
                    size=10,
                    act='softmax',
                    main_program=main_program,
                    startup_program=startup_program,
                    param_attr=param_attr)

label = layers.data(
    name='y',
    shape=[1],
    data_type='int64',
    main_program=main_program,
    startup_program=startup_program)

cost = layers.cross_entropy(
    input=predict,
    label=label,
    main_program=main_program,
    startup_program=startup_program)
avg_cost = layers.mean(
    x=cost, main_program=main_program, startup_program=startup_program)
accuracy = layers.accuracy(
    input=predict,
    label=label,
    main_program=main_program,
    startup_program=startup_program)

optimizer = optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
opts = optimizer.minimize(avg_cost, startup_program)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(startup_program, feed={}, fetch_list=[])

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

        outs = exe.run(main_program,
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost, accuracy])
        out = np.array(outs[0])
        acc = np.array(outs[1])
        if out[0] < 5.0:
            exit(0)  # if avg cost less than 5.0, we think our code is good.
exit(1)
