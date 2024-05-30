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
# model: ppcls^configs^ImageNet^ShuffleNet^ShuffleNetV2_x2_0
# api:paddle.tensor.manipulation.split
from base import *  # noqa: F403

from paddle.static import InputSpec


class SplitCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [11, 976, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1, var_2 = paddle.tensor.manipulation.split(
            var_0, num_or_sections=[488, 488], axis=1
        )
        return var_2, var_1


class TestSplit(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[11, 976, 7, 7], dtype=paddle.float32),
        )
        self.net = SplitCase


if __name__ == '__main__':
    unittest.main()
