# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import time
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_func
from paddle.fluid.dygraph.nn import BatchNorm, Conv2D, Linear, Pool2D

IMAGENET1000 = 1281167
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 8
epoch_num = 1
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() \
    else fluid.CPUPlace()


def optimizer_setting(parameter_list=None):
    total_images = IMAGENET1000
    step = int(math.ceil(float(total_images) / batch_size))
    epochs = [30, 60, 90]
    bd = [step * e for e in epochs]

    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    if fluid.in_dygraph_mode():
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay),
            parameter_list=parameter_list)
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = fluid.layer_helper.LayerHelper(
            self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=102):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))

    @dygraph_to_static_func
    def forward(self, inputs, label):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        pred = self.out(y)

        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss_ = fluid.layers.mean(x=loss)
        acc_top1_ = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5_ = fluid.layers.accuracy(input=pred, label=label, k=5)

        return pred, avg_loss_, acc_top1_, acc_top5_


def train_resnet_in_static_mode():
    """
    Tests model decorated by `dygraph_to_static_output` in static mode. For users, the model is defined in dygraph mode and trained in static mode.
    """

    exe = fluid.Executor(place)
    startup_prog = fluid.Program()
    main_prog = fluid.Program()

    with fluid.program_guard(main_prog, startup_prog):

        img = fluid.data(name="img", shape=[None, 3, 224, 224], dtype="float32")
        label = fluid.data(name="label", shape=[None, 1], dtype="int64")
        label.stop_gradient = True
        resnet = ResNet()
        pred, avg_loss_, acc_top1_, acc_top5_ = resnet(img, label)
        optimizer = optimizer_setting(parameter_list=resnet.parameters())
        optimizer.minimize(avg_loss_)

    exe.run(startup_prog)

    train_reader = paddle.batch(
        paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size)

    for epoch in range(epoch_num):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        for batch_id, data in enumerate(train_reader()):
            start_time = time.time()
            dy_x_data = np.array(
                [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
            if len(np.array([x[1]
                             for x in data]).astype('int64')) != batch_size:
                continue
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1,
                                                                            1)

            avg_loss, acc_top1, acc_top5 = exe.run(
                main_prog,
                feed={"img": dy_x_data,
                      "label": y_data},
                fetch_list=[avg_loss_, acc_top1_, acc_top5_])

            total_loss += avg_loss
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1

            end_time = time.time()
            if batch_id % 2 == 0:
                print( "epoch %d | batch step %d, loss %0.3f, acc1 %0.3f, acc5 %0.3f, time %f" % \
                       ( epoch, batch_id, total_loss / total_sample, \
                         total_acc1 / total_sample, total_acc5 / total_sample, end_time-start_time))
            if batch_id == 10:
                break


class TestResnet(unittest.TestCase):
    def test_in_static_mode(self):
        train_resnet_in_static_mode()


if __name__ == '__main__':
    unittest.main()
