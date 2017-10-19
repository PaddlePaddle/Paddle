import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np

program = Program()
x = layers.data(name='x', shape=[13], data_type='float32', program=program)
y_predict = layers.fc(input=x, size=1, act=None, program=program)

y = layers.data(name='y', shape=[1], data_type='float32', program=program)

cost = layers.square_error_cost(input=y_predict, label=y, program=program)
avg_cost = layers.mean(x=cost, program=program)

sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
opts = sgd_optimizer.minimize(avg_cost)

print str(program)

import pdb
pdb.set_trace()

BATCH_SIZE = 100

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

PASS_NUM = 200
for pass_id in range(PASS_NUM):
    for data in train_reader():
        x_data = np.array(map(lambda x: x[0], data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("float32")
        #y_data = np.expand_dims(y_data, axis=1)

        tensor_x = core.LoDTensor()
        tensor_x.set(x_data, place)

        tensor_y = core.LoDTensor()
        tensor_y.set(y_data, place)
        outs = exe.run(program,
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost])
        out = np.array(outs[0])
        print out
