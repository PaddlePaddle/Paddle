import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor

import numpy as np

startup_program = Program()
main_program = Program()

images = layers.data(
    name='pixel',
    shape=[1, 28, 28],
    data_type='float32',
    main_program=main_program,
    startup_program=startup_program)
label = layers.data(
    name='label',
    shape=[1],
    data_type='int64',
    main_program=main_program,
    startup_program=startup_program)
conv_pool_1 = nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act="relu",
    main_program=main_program,
    startup_program=startup_program)
conv_pool_2 = nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act="relu",
    main_program=main_program,
    startup_program=startup_program)

predict = layers.fc(input=conv_pool_2,
                    size=10,
                    act="softmax",
                    main_program=main_program,
                    startup_program=startup_program)
cost = layers.cross_entropy(
    input=predict,
    label=label,
    main_program=main_program,
    startup_program=startup_program)
avg_cost = layers.mean(x=cost, main_program=main_program)
accuracy = layers.accuracy(
    input=predict,
    label=label,
    main_program=main_program,
    startup_program=startup_program)

# optimizer = optimizer.MomentumOptimizer(learning_rate=0.1 / 128.0,
# momentum=0.9)
optimizer = optimizer.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
opts = optimizer.minimize(avg_cost, startup_program)

BATCH_SIZE = 50
PASS_NUM = 3
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(startup_program, feed={}, fetch_list=[])

for pass_id in range(PASS_NUM):
    count = 0
    for data in train_reader():
        img_data = np.array(map(lambda x: x[0].reshape([1, 28, 28]),
                                data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([BATCH_SIZE, 1])

        tensor_img = core.LoDTensor()
        tensor_y = core.LoDTensor()
        tensor_img.set(img_data, place)
        tensor_y.set(y_data, place)

        outs = exe.run(main_program,
                       feed={"pixel": tensor_img,
                             "label": tensor_y},
                       fetch_list=[avg_cost, accuracy])
        loss = np.array(outs[0])
        acc = np.array(outs[1])

        if loss < 10.0 and acc > 0.9:
            # if avg cost less than 10.0 and accuracy is larger than 0.9, we think our code is good.
            exit(0)
exit(1)
