# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleDetection
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:reshape||api:paddle.nn.functional.activation.softmax||method:transpose||api:paddle.nn.functional.conv._conv_nd||method:squeeze||api:paddle.tensor.manipulation.split||method:__neg__||method:__add__||method:__add__||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1, 17, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [2541, 2], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 2541, 68], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2, var_3, var_4 = var_1.shape
        var_5 = var_1.reshape([-1, var_3, 4, 17])
        var_6 = paddle.nn.functional.activation.softmax(var_5)
        var_7 = var_6.transpose([0, 3, 1, 2])
        var_8 = paddle.nn.functional.conv._conv_nd(
            var_7,
            self.parameter_0,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_9 = var_8.squeeze(1)
        var_10, var_11 = paddle.tensor.manipulation.split(var_9, 2, -1)
        var_12 = -var_10
        var_13 = var_12 + var_0
        var_14 = var_11 + var_0
        var_15 = paddle.tensor.manipulation.concat([var_13, var_14], -1)
        return var_15


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[2541, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 2541, 68], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
