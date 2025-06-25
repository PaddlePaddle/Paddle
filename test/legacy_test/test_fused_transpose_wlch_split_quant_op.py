#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

import paddle
import paddle.incubate.nn.functional as F


def fused_transpose_wlch_split_quant_ref(x, tokens_per_expert, pow_2_scales):
    W, L, C, H = x.shape

    x = x.transpose([3, 1, 0, 2]).reshape([H, L * W * C // 128, 128])
    amax = x.astype('float32').abs().max(axis=-1)

    fp8_max = paddle.to_tensor(448.0, 'float32')
    scale = paddle.divide(fp8_max, amax)
    if pow_2_scales:
        _, exp = paddle.frexp(scale)
        scale = paddle.ldexp(paddle.to_tensor([1.0]), exp - 1)
    one = paddle.to_tensor(1.0, 'float32')
    scale = paddle.where(amax == 0, one, scale)

    out = x * scale.unsqueeze(2)
    out = out.reshape([H, L * W * C]).astype('float8_e4m3fn')
    outs = out.split(tokens_per_expert, axis=1)

    scale = paddle.reciprocal(scale).transpose([1, 0])
    scales = scale.split([i // 128 for i in tokens_per_expert], axis=0)

    return outs, scales


class TestFusedTransposeWLCHSplitQuantOp(unittest.TestCase):
    def eval(self, shape, tokens_per_expert, pow_2_scales):
        x = paddle.randn(shape, 'bfloat16').clip(-50, 50)
        out, scale = F.fused_transpose_wlch_split_quant(
            x, tokens_per_expert, pow_2_scales
        )
        out_ref, scale_ref = fused_transpose_wlch_split_quant_ref(
            x, tokens_per_expert, pow_2_scales
        )
        for a, b in zip(out, out_ref):
            np.testing.assert_allclose(
                a.astype('float32'), b.astype('float32'), atol=0, rtol=0
            )
        for a, b in zip(scale, scale_ref):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)

    def test_0_size_expert(self):
        self.eval(
            shape=[0, 1, 512, 1024],
            tokens_per_expert=[0, 0, 0, 0],
            pow_2_scales=False,
        )

    def test_0_size_token(self):
        self.eval(
            shape=[4, 8, 32, 0],
            tokens_per_expert=[384, 128, 0, 512],
            pow_2_scales=True,
        )

    def test_small(self):
        self.eval(
            shape=[4, 32, 1, 49],
            tokens_per_expert=[128],
            pow_2_scales=False,
        )

    def test_medium(self):
        self.eval(
            shape=[4, 8, 32, 102],
            tokens_per_expert=[384, 128, 0, 512],
            pow_2_scales=True,
        )

    def test_large(self):
        self.eval(
            shape=[5, 21, 128, 256],
            tokens_per_expert=[0, 896, 2432, 1024, 4992, 0, 3968, 128],
            pow_2_scales=False,
        )


if __name__ == "__main__":
    unittest.main()
