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
# model: ppcls^configs^ImageNet^GhostNet^GhostNet_x0_5
# api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class ConcatCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [145, 12, 112, 112], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [145, 12, 112, 112], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.manipulation.concat([var_1, var_0], axis=1)
        return var_2


class TestConcat(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[145, 12, 112, 112], dtype=paddle.float32),
            paddle.rand(shape=[145, 12, 112, 112], dtype=paddle.float32),
        )
        self.net = ConcatCase


if __name__ == '__main__':
    unittest.main()
