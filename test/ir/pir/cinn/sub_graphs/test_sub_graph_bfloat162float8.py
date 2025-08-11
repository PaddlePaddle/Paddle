# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# repo: CINN support float8e4m3

from base import *  # noqa: F403

import paddle
from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        # (shape: [8 * 4096, 2048 * 2, dtype: paddle.bfloat16, stop_gradient: False)
        x,
        # (shape: [8 * 4096, 1], dtype: paddle.float32, stop_gradient: False)
        prob,
    ):
        o2_p = (
            x * prob
        )  # single_op_fallback_to_phi Pass will fallback CastOp to PHI
        o2_p_fp8 = o2_p.astype(paddle.float8_e4m3fn)  # bf16 -> fp8
        o2_p_fp8.stop_gradient = True
        return o2_p_fp8


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(8 * 4096, 2048 * 2),
                dtype=paddle.bfloat16,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(8 * 4096, 1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]

        self.inputs = (
            paddle.clip(
                paddle.randn([8 * 4096, 2048 * 2]).astype("bfloat16"),
                min=-50,
                max=50,
            ),
            paddle.rand(shape=[8 * 4096, 1], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-8


if __name__ == '__main__':
    unittest.main()
