# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
import unittest

from test_trt_explicit_quantization_model import TestExplicitQuantizationModel

import paddle


class ResNet:
    def __init__(self, layers=50, prefix_name=''):
        self.layers = layers
        self.prefix_name = prefix_name

    def net(self, input, class_dim=1000, conv1_name='conv1', fc_name=None):
        layers = self.layers
        prefix_name = (
            self.prefix_name
            if self.prefix_name == ''
            else self.prefix_name + '_'
        )
        supported_layers = [34, 50, 101, 152]
        assert (
            layers in supported_layers
        ), f"supported layers are {supported_layers} but input layer is {layers}"

        if layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name=prefix_name + conv1_name,
        )

        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv_name = prefix_name + conv_name
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                    )

            pool = paddle.nn.functional.adaptive_avg_pool2d(conv, 1)
            pool = paddle.reshape(pool, [-1, 2048])
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)

            fc_name = fc_name if fc_name is None else prefix_name + fc_name
            out = paddle.static.nn.fc(
                pool,
                class_dim,
                activation='softmax',
                name=fc_name,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
                ),
            )
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv_name = prefix_name + conv_name
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name,
                    )

            pool = paddle.nn.functional.adaptive_avg_pool2d(conv, 1)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            fc_name = fc_name if fc_name is None else prefix_name + fc_name
            out = paddle.static.nn.fc(
                pool,
                class_dim,
                name=fc_name,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
                ),
            )

        return out

    def conv_bn_layer(
        self,
        input,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
    ):
        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=paddle.ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
        )
        if self.prefix_name == '':
            if name == "conv1":
                bn_name = "bn_" + name
            else:
                bn_name = "bn" + name[3:]
        else:
            if name.split("_")[1] == "conv1":
                bn_name = name.split("_", 1)[0] + "_bn_" + name.split("_", 1)[1]
            else:
                bn_name = (
                    name.split("_", 1)[0] + "_bn" + name.split("_", 1)[1][3:]
                )
        return paddle.static.nn.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=paddle.ParamAttr(name=bn_name + '_scale'),
            bias_attr=paddle.ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
        )

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
        )
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
        )
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
        )

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1",
        )

        out = paddle.add(x=short, y=conv2, name=name + ".add.output.5")
        return paddle.nn.functional.relu(out)

    def basic_block(self, input, num_filters, stride, is_first, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a",
        )
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b",
        )
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1"
        )

        out = paddle.add(x=short, y=conv1)
        return paddle.nn.functional.relu(out)


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 5, 1),
    "Quantization axis is consistent with Paddle after TRT 8.5.2.",
)
class TestExplicitQuantizationResNet(
    TestExplicitQuantizationModel, unittest.TestCase
):
    def build_model(self):
        model = ResNet(layers=50, prefix_name='')
        return model


if __name__ == '__main__':
    unittest.main()
