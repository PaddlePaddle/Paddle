import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program
from paddle.v2.framework.io import save_persistables, load_persistables
from paddle.v2.framework.executor import Executor

import numpy as np

startup_program = Program()
main_program = Program()
x = layers.data(
    name='x',
    shape=[13],
    data_type='float32',
    main_program=main_program,
    startup_program=startup_program)

y_predict = layers.fc(input=x,
                      size=1,
                      act=None,
                      main_program=main_program,
                      startup_program=startup_program)

y = layers.data(
    name='y',
    shape=[1],
    data_type='float32',
    main_program=main_program,
    startup_program=startup_program)

cost = layers.square_error_cost(
    input=y_predict,
    label=y,
    main_program=main_program,
    startup_program=startup_program)
avg_cost = layers.mean(
    x=cost, main_program=main_program, startup_program=startup_program)

sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
opts = sgd_optimizer.minimize(avg_cost, startup_program)

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(startup_program, feed={}, fetch_list=[])

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    save_persistables(exe, "./fit_a_line.model/", main_program=main_program)
    load_persistables(exe, "./fit_a_line.model/", main_program=main_program)
    for data in train_reader():
        x_data = np.array(map(lambda x: x[0], data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("float32")

        tensor_x = core.LoDTensor()
        tensor_x.set(x_data, place)
        # print tensor_x.get_dims()

        tensor_y = core.LoDTensor()
        tensor_y.set(y_data, place)
        # print tensor_y.get_dims()
        outs = exe.run(main_program,
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost])
        out = np.array(outs[0])

        if out[0] < 10.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
