import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

x = fluid.layers.data(name='x', shape=[13], dtype='float32')

y_predict = fluid.layers.fc(input=x, size=1, act=None)

y = fluid.layers.data(name='y', shape=[1], dtype='float32')

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(x=cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    fluid.io.save_persistables(exe, "./fit_a_line.model/")
    fluid.io.load_persistables(exe, "./fit_a_line.model/")
    for data in train_reader():
        avg_loss_value, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])

        if avg_loss_value[0] < 10.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
