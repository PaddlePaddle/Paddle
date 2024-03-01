#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base

paddle.enable_static()

# Fix seed for test
paddle.seed(1)

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001],
    },
}


class SE_ResNeXt:
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), f"supported layers are {supported_layers} but input layer is {layers}"
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input, num_filters=64, filter_size=7, stride=2, act='relu'
            )
            conv = paddle.nn.functional.max_pool2d(
                x=conv,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input, num_filters=64, filter_size=7, stride=2, act='relu'
            )
            conv = paddle.nn.functional.max_pool2d(
                x=conv,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input, num_filters=64, filter_size=3, stride=2, act='relu'
            )
            conv = self.conv_bn_layer(
                input=conv, num_filters=64, filter_size=3, stride=1, act='relu'
            )
            conv = self.conv_bn_layer(
                input=conv, num_filters=128, filter_size=3, stride=1, act='relu'
            )
            conv = paddle.nn.functional.max_pool2d(
                x=conv,
                kernel_size=3,
                stride=2,
                padding=1,
            )

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                )

        pool = paddle.nn.functional.adaptive_avg_pool2d(x=conv, output_size=1)
        drop = paddle.nn.functional.dropout(x=pool, p=0.2)

        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = paddle.static.nn.fc(
            x=drop,
            size=class_dim,
            activation='softmax',
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
        )
        return out

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(input, ch_out, filter_size, stride)
        else:
            return input

    def bottleneck_block(
        self, input, num_filters, stride, cardinality, reduction_ratio
    ):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu'
        )
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
        )
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 2, filter_size=1, act=None
        )
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
        )

        short = self.shortcut(input, num_filters * 2, stride)

        return paddle.nn.functional.relu(paddle.add(x=short, y=scale))

    def conv_bn_layer(
        self, input, num_filters, filter_size, stride=1, groups=1, act=None
    ):
        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            # avoid pserver CPU init differs from GPU
            param_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
            bias_attr=False,
        )
        return paddle.static.nn.batch_norm(input=conv, act=act)

    def squeeze_excitation(self, input, num_channels, reduction_ratio):
        pool = paddle.nn.functional.adaptive_avg_pool2d(x=input, output_size=1)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = paddle.static.nn.fc(
            x=pool,
            size=num_channels // reduction_ratio,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
            activation='relu',
        )
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = paddle.static.nn.fc(
            x=squeeze,
            size=num_channels,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.05)
            ),
            activation='sigmoid',
        )
        scale = paddle.tensor.math._multiply_with_axis(
            x=input, y=excitation, axis=0
        )
        return scale


class DistSeResneXt2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False):
        # Input data
        image = paddle.static.data(
            name="data", shape=[-1, 3, 224, 224], dtype='float32'
        )
        label = paddle.static.data(name="int64", shape=[-1, 1], dtype='int64')

        # Train program
        model = SE_ResNeXt(layers=50)
        out = model.net(input=image, class_dim=102)
        cost = paddle.nn.functional.cross_entropy(
            input=out, label=label, reduction='none', use_softmax=True
        )

        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.static.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.static.accuracy(input=out, label=label, k=5)

        # Evaluator
        test_program = base.default_main_program().clone(for_test=True)

        # Optimization
        total_images = 6149  # flowers
        epochs = [30, 60, 90]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in epochs]
        base_lr = 0.1
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]

        if not use_dgc:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                    boundaries=bd, values=lr
                ),
                momentum=0.9,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
            )
        else:
            optimizer = (
                paddle.distributed.fleet.meta_optimizers.DGCMomentumOptimizer(
                    learning_rate=paddle.optimizer.lr.piecewise_decay(
                        boundaries=bd, values=lr
                    ),
                    momentum=0.9,
                    rampup_begin_step=0,
                    regularization=paddle.regularizer.L2Decay(1e-4),
                )
            )
        optimizer.minimize(avg_cost)

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False), batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False), batch_size=batch_size
        )

        return test_program, avg_cost, train_reader, test_reader, acc_top1, out


if __name__ == "__main__":
    runtime_main(DistSeResneXt2x2)
