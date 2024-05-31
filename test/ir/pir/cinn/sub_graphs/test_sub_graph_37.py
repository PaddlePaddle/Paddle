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
# model: ppcls^configs^ImageNet^CSWinTransformer^CSWinTransformer_base_384
# api:paddle.nn.functional.norm.layer_norm||api:paddle.tensor.stat.mean
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [4, 144, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.norm.layer_norm(
            var_0,
            normalized_shape=[768],
            weight=self.parameter_0,
            bias=self.parameter_1,
            epsilon=1e-05,
        )
        var_2 = paddle.tensor.stat.mean(var_1, axis=1)
        return var_2


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[4, 144, 768], dtype=paddle.float32),)
        self.net = LayerCase


# if __name__ == '__main__':
#     unittest.main()
