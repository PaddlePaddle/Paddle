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
# model: ppcls^configs^ImageNet^ResNet^ResNet50_amp_O2_ultra
# api:paddle.tensor.linalg.transpose
from base import *  # noqa: F403

from paddle.static import InputSpec


class TransposeCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [22, 4, 224, 224], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.tensor.linalg.transpose(var_0, [0, 2, 3, 1])
        return var_1


class TestTranspose(TestBase):
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
            paddle.rand(shape=[22, 4, 224, 224], dtype=paddle.float32),
        )
        self.net = TransposeCase


if __name__ == '__main__':
    unittest.main()
