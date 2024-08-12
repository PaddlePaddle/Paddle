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
# api:paddle.tensor.creation.to_tensor||method:reshape||method:transpose||method:reshape
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [12, 288, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = var_0.reshape([6, 2, 1, 12, 24, 192])
        var_3 = var_2.transpose([0, 1, 3, 2, 4, 5])
        var_4 = var_3.reshape([6, 24, 24, 192])
        return var_4


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
        self.inputs = (paddle.rand(shape=[12, 288, 192], dtype=paddle.float32),)
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
