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
# model: deepseek v3
# api:paddle.nn.functional.conv._conv_nd||method:reshape||method:unsqueeze||api:paddle.nn.functional.common.unfold||method:reshape||method:__mul__||method:sum||method:reshape
from typing import Tuple  # noqa: UP035

from base import *  # noqa: F403

import paddle
import paddle.incubate.nn.functional as F
from paddle import Tensor
from paddle.static import InputSpec


def act_quant(x: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: UP006
    """
    Quantizes input tensor to FP8-E4M3 format with dynamic scaling.
    Uses reshape instead of view for better compatibility.
    Args:
        x: Input tensor of shape (m, n) where n % 128 == 0
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, n = x.shape

    # Ensure tensor is contiguous before reshaping
    if not x.is_contiguous():
        x = x.contiguous()

    # Reshape input for processing
    x_reshaped = paddle.reshape(
        x, (m, -1, 128)
    )  # [8 * 4096, 2048]->[8 * 4096, 16, 128]

    # Calculate scale factors in float32
    x_abs = paddle.abs(x_reshaped).astype(paddle.float32)
    x_amax = paddle.max(x_abs, axis=2)  #  [8 * 4096, 16, 128] -> [8 * 4096, 16]
    x_amax = paddle.reshape(x_amax, (m, -1))  # [8 * 4096, 16] = 524288
    x_amax = paddle.clip(x_amax, min=1e-4)

    # Calculate power-of-2 scale factors
    scale = x_amax.unsqueeze(2) / 448.0  # [8 * 4096, 16, 1] = 524288
    two = paddle.to_tensor(2.0, dtype=scale.dtype)
    scale = paddle.pow(two, paddle.ceil(paddle.log2(scale)))

    # Quantize and clip
    scaled_x = x_reshaped / scale  # [8*4096, 16, 128]
    scaled_x = paddle.clip(scaled_x, min=-448, max=448)

    # Convert to FP8 and reshape final output
    quantized_x = paddle.reshape(
        scaled_x.astype(paddle.float8_e4m3fn), (m, n)
    )  # [8 * 4096, 2048]
    scale_factors = paddle.reshape(scale, (m, -1))  # [8 * 4096, 16]

    return (quantized_x, scale_factors)


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
        o2 = F.swiglu(x)  # bf16 -> bf16 [8 * 4096, 2048 * 2]->[8 * 4096, 2048]
        o2_p = (
            o2 * prob
        )  # bf16 * fp32 -> bf16 [8 * 4096, 2048] * [8 * 4096, 1]-> [8 * 4096, 2048]
        o2_p_fp8, t2_scale = act_quant(o2_p)  # bf16 -> fp8, fp32
        o2_p_fp8.stop_gradient = True
        t2_scale.stop_gradient = True
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
        self.atol = 1e-5


if __name__ == '__main__':
    unittest.main()
