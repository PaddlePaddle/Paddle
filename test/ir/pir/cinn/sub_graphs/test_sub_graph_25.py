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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^EfficientNet^EfficientNetB0
# api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.common.dropout||api:paddle.tensor.manipulation.squeeze||api:paddle.nn.functional.common.linear
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1280, 1000],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [11, 1280, 7, 7], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_0, output_size=1, data_format='NCHW', name=None
        )
        var_2 = paddle.nn.functional.common.dropout(
            var_1,
            p=0.2,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_3 = paddle.tensor.manipulation.squeeze(var_2, axis=[2, 3])
        var_4 = paddle.nn.functional.common.linear(
            x=var_3, weight=self.parameter_0, bias=self.parameter_1, name=None
        )
        return var_4


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[11, 1280, 7, 7], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
