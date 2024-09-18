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

import unittest

from test_trt_explicit_quantization_model import TestExplicitQuantizationModel

import paddle
from paddle.nn.initializer import KaimingUniform


class MobileNet:
    def __init__(self):
        self.params = {
            "input_size": [3, 224, 224],
            "input_mean": [0.485, 0.456, 0.406],
            "input_std": [0.229, 0.224, 0.225],
            "learning_strategy": {
                "name": "piecewise_decay",
                "batch_size": 256,
                "epochs": [10, 16, 30],
                "steps": [0.1, 0.01, 0.001, 0.0001],
            },
        }

    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1",
        )

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale,
            name="conv2_2",
        )

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale,
            name="conv3_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale,
            name="conv3_2",
        )

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale,
            name="conv4_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale,
            name="conv4_2",
        )

        # 14x14
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale,
                name="conv5" + "_" + str(i + 1),
            )
        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=2,
            scale=scale,
            name="conv5_6",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale,
            name="conv6",
        )

        input = paddle.nn.functional.adaptive_avg_pool2d(input, 1)
        with paddle.static.name_scope('last_fc'):
            output = paddle.static.nn.fc(
                input,
                class_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=KaimingUniform(), name="fc7_weights"
                ),
                bias_attr=paddle.ParamAttr(name="fc7_offset"),
            )

        return output

    def conv_bn_layer(
        self,
        input,
        filter_size,
        num_filters,
        stride,
        padding,
        channels=None,
        num_groups=1,
        act='relu',
        use_cudnn=True,
        name=None,
    ):

        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=paddle.ParamAttr(
                initializer=KaimingUniform(), name=name + "_weights"
            ),
            bias_attr=False,
        )
        bn_name = name + "_bn"
        return paddle.static.nn.batch_norm(
            input=conv,
            act=act,
            param_attr=paddle.ParamAttr(name=bn_name + "_scale"),
            bias_attr=paddle.ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
        )

    def depthwise_separable(
        self,
        input,
        num_filters1,
        num_filters2,
        num_groups,
        stride,
        scale,
        name=None,
    ):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw",
        )

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep",
        )
        return pointwise_conv


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 5, 1),
    "Quantization axis is consistent with Paddle after TRT 8.5.2.",
)
class TestExplicitMobilenet(TestExplicitQuantizationModel, unittest.TestCase):
    def build_model(self):
        model = MobileNet()
        return model


if __name__ == '__main__':
    unittest.main()
