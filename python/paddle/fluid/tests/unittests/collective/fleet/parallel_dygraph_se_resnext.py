# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import math

import numpy as np
from test_dist_base import TestParallelDyGraphRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.nn import Linear

batch_size = 64
momentum_rate = 0.9
l2_decay = 1.2e-4

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "cosine_decay",
        "batch_size": batch_size,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001],
    },
    "batch_size": batch_size,
    "lr": 0.0125,
    "total_images": 6149,
    "num_epochs": 200,
}


def optimizer_setting(params, parameter_list=None):
    ls = params["learning_strategy"]
    if "total_images" not in params:
        total_images = 6149
    else:
        total_images = params["total_images"]

    batch_size = ls["batch_size"]
    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    if fluid._non_static_mode():
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs
            ),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay),
            parameter_list=parameter_list,
        )
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs
            ),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay),
        )

    return optimizer


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
        )

        # disable BatchNorm in multi-card. disable LayerNorm because of complex input_shape
        # self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        # y = self._batch_norm(y)

        return y


class SqueezeExcitation(fluid.dygraph.Layer):
    def __init__(self, num_channels, reduction_ratio):

        super().__init__()
        self._num_channels = num_channels
        self._pool = paddle.nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self._squeeze = Linear(
            num_channels,
            num_channels // reduction_ratio,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )
        stdv = 1.0 / math.sqrt(num_channels / 16.0 * 1.0)
        self._excitation = Linear(
            num_channels // reduction_ratio,
            num_channels,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, input):
        y = self._pool(input)
        y = paddle.reshape(y, shape=[-1, self._num_channels])
        y = self._squeeze(y)
        y = paddle.nn.functional.relu(y)
        y = self._excitation(y)
        y = paddle.nn.functional.sigmoid(y)
        y = paddle.tensor.math._multiply_with_axis(x=input, y=y, axis=0)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        cardinality,
        reduction_ratio,
        shortcut=True,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act="relu",
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
        )

        self.scale = SqueezeExcitation(
            num_channels=num_filters * 2, reduction_ratio=reduction_ratio
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2,
                filter_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 2

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.nn.functional.relu(paddle.add(x=short, y=scale))
        return y


class SeResNeXt(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=102):
        super().__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
            )
            self.conv1 = ConvBNLayer(
                num_channels=64,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
            )
            self.conv2 = ConvBNLayer(
                num_channels=64,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut,
                    ),
                )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 2 * 1 * 1

        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, inputs):
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            y = self.pool(y)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(inputs)
            y = self.conv2(inputs)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y


class TestSeResNeXt(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SeResNeXt()
        train_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False),
            batch_size=train_parameters["batch_size"],
            drop_last=True,
        )
        optimizer = optimizer_setting(
            train_parameters, parameter_list=model.parameters()
        )
        return model, train_reader, optimizer

    def run_one_loop(self, model, opt, data):
        bs = len(data)
        dy_x_data = np.array([x[0].reshape(3, 224, 224) for x in data]).astype(
            'float32'
        )
        dy_x_data = dy_x_data / 255.0
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(bs, 1)
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True

        out = model(img)
        softmax_out = paddle.nn.functional.softmax(out, use_cudnn=False)
        loss = paddle.nn.functional.cross_entropy(
            input=softmax_out, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(x=loss)
        return avg_loss


if __name__ == "__main__":
    runtime_main(TestSeResNeXt)
