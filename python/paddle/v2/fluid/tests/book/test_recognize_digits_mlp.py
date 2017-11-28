from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 128
image = fluid.layers.data(name='x', shape=[784], dtype='float32')

param_attr = {
    'name': None,
    'regularization': fluid.regularizer.L2Decay(0.0005 * BATCH_SIZE)
}

hidden1 = fluid.layers.fc(input=image,
                          size=128,
                          act='relu',
                          param_attr=param_attr)
hidden2 = fluid.layers.fc(input=hidden1,
                          size=64,
                          act='relu',
                          param_attr=param_attr)

predict = fluid.layers.fc(input=hidden2,
                          size=10,
                          act='softmax',
                          param_attr=param_attr)

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

exe.run(fluid.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    for data in train_reader():
        x_data = np.array(map(lambda x: x[0], data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = np.expand_dims(y_data, axis=1)

        tensor_x = fluid.LoDTensor()
        tensor_x.set(x_data, place)

        tensor_y = fluid.LoDTensor()
        tensor_y.set(y_data, place)

        outs = exe.run(fluid.default_main_program(),
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost] + accuracy.metrics)
        out = np.array(outs[0])
        acc = np.array(outs[1])
        pass_acc = accuracy.eval(exe)

        test_accuracy.reset(exe)
        for data in test_reader():
            x_data = np.array(map(lambda x: x[0], data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = np.expand_dims(y_data, axis=1)

            out, acc = exe.run(inference_program,
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_cost] + test_accuracy.metrics)

        test_pass_acc = test_accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " train_cost=" + str(
            out) + " train_acc=" + str(acc) + " train_pass_acc=" + str(pass_acc)
              + " test_acc=" + str(test_pass_acc))

        if test_pass_acc > 0.7:
            exit(0)
exit(1)
