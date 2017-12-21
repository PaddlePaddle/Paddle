from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import os

images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
conv_pool_1 = fluid.nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act="relu")
conv_pool_2 = fluid.nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act="relu")

predict = fluid.layers.fc(input=conv_pool_2, size=10, act="softmax")
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)
optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimize_ops, params_grads = optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

BATCH_SIZE = 50
PASS_NUM = 3
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
t = fluid.DistributeTranspiler()
pserver_endpoints = os.getenv("PSERVERS")
training_role = os.getenv("TRAINING_ROLE",
                          "TRAINER")  # get the training role: trainer/pserver
t.transpile(optimize_ops, params_grads, pservers=pserver_endpoints, trainers=1)

if training_role == "PSERVER":
    pserver_prog = t.get_pserver_program(pserver_endpoints, optimize_ops)
    exe.run(fluid.default_startup_program())
    exe.run(pserver_prog)
elif training_role == "TRAINER":
    feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
    exe.run(fluid.default_startup_program())

    for pass_id in range(PASS_NUM):
        accuracy.reset(exe)
        for data in train_reader():
            loss, acc = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)
            # print loss, acc
            if loss < 10.0 and pass_acc > 0.9:
                # if avg cost less than 10.0 and accuracy is larger than 0.9, we think our code is good.
                exit(0)

        pass_acc = accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc))
else:
    print("environment var TRAINER_ROLE should be TRAINER os PSERVER")

exit(1)
