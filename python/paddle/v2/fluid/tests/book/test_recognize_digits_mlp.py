import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.evaluator as evaluator
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.initializer import UniformInitializer
from paddle.v2.fluid.optimizer import MomentumOptimizer
from paddle.v2.fluid.regularizer import L2DecayRegularizer

BATCH_SIZE = 128
image = layers.data(name='x', shape=[784], data_type='float32')

param_attr = {
    'name': None,
    'initializer': UniformInitializer(
        low=-1.0, high=1.0),
    'regularization': L2DecayRegularizer(0.0005 * BATCH_SIZE)
}

hidden1 = layers.fc(input=image, size=128, act='relu', param_attr=param_attr)
hidden2 = layers.fc(input=hidden1, size=64, act='relu', param_attr=param_attr)

predict = layers.fc(input=hidden2,
                    size=10,
                    act='softmax',
                    param_attr=param_attr)

label = layers.data(name='y', shape=[1], data_type='int64')

cost = layers.cross_entropy(input=predict, label=label)
avg_cost = layers.mean(x=cost)

optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
opts = optimizer.minimize(avg_cost)

accuracy, acc_out = evaluator.accuracy(input=predict, label=label)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(framework.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    for data in train_reader():
        x_data = np.array(map(lambda x: x[0], data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = np.expand_dims(y_data, axis=1)

        tensor_x = core.LoDTensor()
        tensor_x.set(x_data, place)

        tensor_y = core.LoDTensor()
        tensor_y.set(y_data, place)

        outs = exe.run(framework.default_main_program(),
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost, acc_out])
        out = np.array(outs[0])
        acc = np.array(outs[1])
        pass_acc = accuracy.eval(exe)

        if pass_acc > 0.7:
            exit(0)
            # print("pass_id=" + str(pass_id) + " auc=" +
            #      str(acc) + " pass_acc=" + str(pass_acc))
exit(1)
