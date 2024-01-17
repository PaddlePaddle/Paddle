#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
from paddle import _C_ops
from paddle.base import core


def composite_rms_norm(x, scale, epsilon=1e-5):
    hidden_states = x.astype("float32")
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = paddle.rsqrt(variance + epsilon) * hidden_states
    if scale.dtype in [paddle.float16, paddle.bfloat16]:
        hidden_states = paddle.cast(hidden_states, scale.dtype)
    return hidden_states * scale


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA ",
)
class TestFusedRmsNorm(unittest.TestCase):
    def setUp(self):
        pass

    def get_paddle_tensor(self, shape, dtype, bound=0.5):
        tmp = paddle.uniform(shape, dtype=dtype, min=-bound, max=bound)
        tmp.stop_gradient = False
        return tmp

    def get_forward_backward(self, func, seed, dtype):
        paddle.disable_static()
        paddle.seed(seed)
        x = self.get_paddle_tensor([2, 256], dtype)
        scale = self.get_paddle_tensor([256], dtype)
        out_g = paddle.randn([2, 256], dtype)
        out = func(x, scale, 1e-5)
        paddle.autograd.backward([out], [out_g], True)
        return out, (x.grad, scale.grad)

    def test_fused_rms_norm(self):
        dtypes = [paddle.float32]
        if paddle.amp.is_bfloat16_supported('gpu'):
            dtypes.append(paddle.bfloat16)
        if paddle.amp.is_float16_supported('gpu'):
            dtypes.append(paddle.float16)
        for dtype in dtypes:
            raw_out, raw_grads = self.get_forward_backward(
                composite_rms_norm, seed=2024, dtype=dtype
            )
            fused_out, fused_grads = self.get_forward_backward(
                _C_ops.fused_rms_norm, seed=2024, dtype=dtype
            )
            # forward rtol
            rtol = 1e-5 if dtype == paddle.float32 else 1e-3
            np.testing.assert_allclose(
                raw_out.astype(paddle.float32).numpy(),
                fused_out.astype(paddle.float32).numpy(),
                rtol=rtol,
            )
            # backward rtol, only check float32 grad
            rtol = 1e-3
            if dtype == paddle.float32:
                raw_x_grad, raw_scale_grad = raw_grads
                fused_x_grad, fused_scale_grad = fused_grads
                np.testing.assert_allclose(
                    raw_x_grad.astype(paddle.float32).numpy(),
                    fused_x_grad.astype(paddle.float32).numpy(),
                    rtol=rtol,
                )
                np.testing.assert_allclose(
                    raw_scale_grad.astype(paddle.float32).numpy(),
                    fused_scale_grad.astype(paddle.float32).numpy(),
                    rtol=rtol,
                )


if __name__ == '__main__':
    unittest.main()
