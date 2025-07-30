#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.base.param_attr import ParamAttr
from paddle.nn import BatchNorm
from paddle.regularizer import L2Decay


class ConvBNLayer(paddle.nn.Layer):
    def __init__(
        self,
        ch_in,
        ch_out,
        filter_size=3,
        stride=1,
        groups=1,
        padding=0,
        act="leaky",
        is_test=True,
    ):
        super().__init__()

        self.conv = paddle.nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.Normal(0.0, 0.02)
            ),
            bias_attr=False,
        )
        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Normal(0.0, 0.02),
                regularizer=L2Decay(0.0),
            ),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0),
                regularizer=L2Decay(0.0),
            ),
        )

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = paddle.nn.functional.leaky_relu(out, 0.1)
        return out


class DownSample(paddle.nn.Layer):
    def __init__(
        self, ch_in, ch_out, filter_size=3, stride=2, padding=1, is_test=True
    ):
        super().__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            is_test=is_test,
        )
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, is_test=True):
        super().__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
        )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
        )

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.add(x=inputs, y=conv2)
        return out


class LayerWarp(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, count, is_test=True):
        super().__init__()

        self.basicblock0 = BasicBlock(ch_in, ch_out, is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer(
                f"basic_block_{i}",
                BasicBlock(ch_out * 2, ch_out, is_test=is_test),
            )
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(paddle.nn.Layer):
    def __init__(self, ch_in=3, is_test=True):
        super().__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
        )

        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2, is_test=is_test)
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                f"stage_{i}",
                LayerWarp(int(ch_in[i]), 32 * (2**i), stage, is_test=is_test),
            )
            self.darknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                f"stage_{i}_downsample",
                DownSample(
                    ch_in=32 * (2 ** (i + 1)),
                    ch_out=32 * (2 ** (i + 2)),
                    is_test=is_test,
                ),
            )
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]
