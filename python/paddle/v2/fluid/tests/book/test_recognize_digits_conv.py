import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.evaluator as evaluator
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import AdamOptimizer
from paddle.v2.fluid.initializer import NormalInitializer
import numpy as np
import time

BATCH_SIZE = 128
PASS_NUM = 5
SEED = 1
DTYPE = "float32"

images = layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
label = layers.data(name='label', shape=[1], dtype='int64')
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

# TODO(dzhwinter) : refine the initializer and random seed settting
SIZE = 10
input_shape = conv_pool_2.shape
param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

predict = layers.fc(input=conv_pool_2,
                    size=SIZE,
                    act="softmax",
                    param_initializer=NormalInitializer(
                        loc=0.0, scale=scale, seed=SEED))

cost = layers.cross_entropy(input=predict, label=label)
avg_cost = layers.mean(x=cost)
optimizer = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
opts = optimizer.minimize(avg_cost)

accuracy, acc_out = evaluator.accuracy(input=predict, label=label)

train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)

place = core.CPUPlace()
exe = Executor(place)

exe.run(framework.default_startup_program())

for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    pass_start = time.clock()
    for batch_id, data in enumerate(train_reader()):
        img_data = np.array(map(lambda x: x[0].reshape([1, 28, 28]),
                                data)).astype(DTYPE)
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([len(y_data), 1])

        tensor_img = core.LoDTensor()
        tensor_y = core.LoDTensor()
        tensor_img.set(img_data, place)
        tensor_y.set(y_data, place)

        start = time.clock()
        outs = exe.run(framework.default_main_program(),
                       feed={"pixel": tensor_img,
                             "label": tensor_y},
                       fetch_list=[avg_cost, acc_out])
        end = time.clock()
        loss = np.array(outs[0])
        acc = np.array(outs[1])
        print "pass=%d, batch=%d, loss=%f, error=%f, elapse=%f" % (
            pass_id, batch_id, loss, 1 - acc, (end - start) / 1000)

        if loss < 10.0 and acc > 0.9:
            # if avg cost less than 10.0 and accuracy is larger than 0.9, we think our code is good.
            exit(0)

    pass_acc = accuracy.eval(exe)
    print "pass=%d, accuracy=%f, elapse=%f" % (pass_id, pass_acc, (
        time.clock() - pass_start) / 1000)
exit(1)
