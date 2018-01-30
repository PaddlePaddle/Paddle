#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

import unittest

import numpy

import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    return fluid.layers.mean(x=loss), fluid.layers.accuracy(
        input=prediction, label=label)


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
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
    return loss_net(conv_pool_2, label)


def main(parallel, nn_type, use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            img_ = pd.read_input(img)
            label_ = pd.read_input(label)
            for o in net_conf(img_, label_):
                pd.write_output(o)

        avg_loss, acc = pd()
        # get mean loss and acc through every devices.
        avg_loss = fluid.layers.mean(x=avg_loss)
        acc = fluid.layers.mean(x=acc)
    else:
        avg_loss, acc = net_conf(img, label)

    test_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    PASS_NUM = 100
    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            # train a mini-batch, fetch nothing
            exe.run(feed=feeder.feed(data))
            if (batch_id + 1) % 10 == 0:
                acc_set = []
                avg_loss_set = []
                for test_data in test_reader():
                    acc_np, avg_loss_np = exe.run(program=test_program,
                                                  feed=feeder.feed(test_data),
                                                  fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val = numpy.array(acc_set).mean()
                avg_loss_val = numpy.array(avg_loss_set).mean()
                if float(acc_val) > 0.85:  # test acc > 85%
                    return
                else:
                    print(
                        'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_val), float(acc_val)))
    assert AssertionError("Recognize Digits model is divergent")


class TestRecognizeDigits(unittest.TestCase):
    pass


def patch_method(parallel, use_cuda, nn_type):
    def __impl__(self):
        main(parallel=parallel, use_cuda=use_cuda, nn_type=nn_type)

    fname = "test_{0}_{1}_{2}".format(nn_type, "cuda"
                                      if use_cuda else "cpu", "parallel"
                                      if parallel else "normal")
    __impl__.__name__ = fname
    setattr(TestRecognizeDigits, fname, __impl__)


for parallel in (True, False):
    for use_cuda in (True, False):
        for nn_type in ('mlp', 'conv'):
            patch_method(parallel, use_cuda, nn_type)

if __name__ == '__main__':
    unittest.main()
