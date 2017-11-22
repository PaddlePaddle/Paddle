import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.evaluator as evaluator
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import AdamOptimizer

images = layers.data(name='pixel', shape=[1, 28, 28], data_type='float32')
label = layers.data(name='label', shape=[1], data_type='int64')
conv_pool_1 = nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act="relu")
conv_pool_2 = nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act="relu")

predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
cost = layers.cross_entropy(input=predict, label=label)
avg_cost = layers.mean(x=cost)
optimizer = AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
opts = optimizer.minimize(avg_cost)

accuracy, acc_out = evaluator.accuracy(input=predict, label=label)

BATCH_SIZE = 50
PASS_NUM = 3
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(framework.default_startup_program())

for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    for data in train_reader():
        img_data = np.array(map(lambda x: x[0].reshape([1, 28, 28]),
                                data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([BATCH_SIZE, 1])

        tensor_img = core.LoDTensor()
        tensor_y = core.LoDTensor()
        tensor_img.set(img_data, place)
        tensor_y.set(y_data, place)

        outs = exe.run(framework.default_main_program(),
                       feed={"pixel": tensor_img,
                             "label": tensor_y},
                       fetch_list=[avg_cost, acc_out])
        loss = np.array(outs[0])
        acc = np.array(outs[1])
        pass_acc = accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " acc=" + str(acc) + " pass_acc=" +
              str(pass_acc))
        # print loss, acc
        if loss < 10.0 and pass_acc > 0.9:
            # if avg cost less than 10.0 and accuracy is larger than 0.9, we think our code is good.
            exit(0)

    pass_acc = accuracy.eval(exe)
    print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc))

exit(1)
