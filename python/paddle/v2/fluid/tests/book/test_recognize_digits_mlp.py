from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 128
image = fluid.layers.data(name='x', shape=[784], dtype='float32')

regularizer = fluid.regularizer.L2Decay(0.0005 * BATCH_SIZE)

hidden1 = fluid.layers.fc(input=image,
                          size=128,
                          act='relu',
                          param_attr=regularizer)
hidden2 = fluid.layers.fc(input=hidden1,
                          size=64,
                          act='relu',
                          param_attr=regularizer)

predict = fluid.layers.fc(input=hidden2,
                          size=10,
                          act='softmax',
                          param_attr=regularizer)

label = fluid.layers.data(name='y', shape=[1], dtype='int64')

cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)

optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
opts = optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

inference_program = fluid.default_main_program().clone()
test_accuracy = fluid.evaluator.Accuracy(
    input=predict, label=label, main_program=inference_program)
test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
inference_program = fluid.io.get_inference_program(
    test_target, main_program=inference_program)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
exe.run(fluid.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    for data in train_reader():
        out, acc = exe.run(fluid.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[avg_cost] + accuracy.metrics)
        pass_acc = accuracy.eval(exe)

        test_accuracy.reset(exe)
        for data in test_reader():
            out, acc = exe.run(inference_program,
                               feed=feeder.feed(data),
                               fetch_list=[avg_cost] + test_accuracy.metrics)

        test_pass_acc = test_accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " train_cost=" + str(
            out) + " train_acc=" + str(acc) + " train_pass_acc=" + str(pass_acc)
              + " test_acc=" + str(test_pass_acc))

        if test_pass_acc > 0.7:
            exit(0)
exit(1)
