import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 20

x = fluid.layers.data(name='x', shape=[13], dtype='float32')

y_predict = fluid.layers.fc(input=x, size=1, act=None)

y = fluid.layers.data(name='y', shape=[1], dtype='float32')

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(x=cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.dataset.uci_housing.test(),
    batch_size=128)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

exe.run(fluid.default_startup_program())

PASS_NUM = 100


def train():
    for pass_id in range(PASS_NUM):
        fluid.io.save_persistables(exe, "./fit_a_line.model/")
        fluid.io.load_persistables(exe, "./fit_a_line.model/")
        for data in train_reader():
            avg_loss_value,  = exe.run(fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost])

            if avg_loss_value[0] < 10.0:
                return  # if avg cost less than 10.0, we think our code is good.


print("Training...")
train()

print("Now performing inference...")
fluid.io.load_persistables(exe, "./fit_a_line.model/")
for data in test_reader():
    out, y_pred, y_label = exe.run(fluid.default_main_program(),
                                   feed=feeder.feed(data),
                                   fetch_list=[avg_cost, y_predict, y])

for i in xrange(len(y_label)):
    print "label=" + str(y_label[i][0]) + ", predict=" + str(y_pred[i][0])
