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

import numpy as np

import paddle


def fused_transpose_split_quant_ref(x, tokens_per_expert, pow_2_scales):
    shape = x.shape
    x = x.reshape([shape[0] // 128, 128, shape[1]])
    amax = x.astype('float32').abs().max(axis=1)

    scale = 448.0 / amax
    if pow_2_scales:
        _, exp = paddle.frexp(scale)
        scale = paddle.ldexp(paddle.to_tensor([1.0]), exp - 1)
    scale = paddle.where(amax == 0, 1.0, scale)

    out = x * scale.unsqueeze(1)
    out = out.reshape(shape).astype('float8_e4m3fn')
    out = out.transpose([1, 0]).split(tokens_per_expert, axis=1)

    scale = paddle.reciprocal(scale)
    scale = scale.split([t // 128 for t in tokens_per_expert], axis=0)
    return out, scale


def test_fused_transpose_split_quant(tokens_per_expert, seq_len, pow_2_scales):

    x = paddle.randn([sum(tokens_per_expert), seq_len], dtype='bfloat16')
    x = paddle.clip(x, min=-50, max=50)

    out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(
        x, tokens_per_expert, pow_2_scales
    )

    out_ref, scale_ref = fused_transpose_split_quant_ref(
        x, tokens_per_expert, pow_2_scales
    )

    for t, t_ref in zip(out, out_ref):
        np.testing.assert_allclose(t.astype('float32'), t_ref.astype('float32'))

    for t, t_ref in zip(scale, scale_ref):
        np.testing.assert_allclose(t, t_ref)


def run():
    test_fused_transpose_split_quant([0, 0], 1024, False)
    test_fused_transpose_split_quant([128, 2 * 128], 0, True)
    test_fused_transpose_split_quant([128], 1, False)
    test_fused_transpose_split_quant([0, 128, 0, 2 * 128], 127, True)
    test_fused_transpose_split_quant([3 * 128, 4 * 128, 5 * 128], 233, False)
    test_fused_transpose_split_quant(
        [24 * 128, 128, 50 * 128, 16 * 128], 2162, True
    )
    test_fused_transpose_split_quant(
        [7 * 128, 29 * 128, 3 * 128, 128 * 128, 13 * 128], 4000, False
    )
    test_fused_transpose_split_quant(
        [18 * 128, 5 * 128, 24 * 128, 128, 6 * 128, 0, 27 * 128, 7 * 128],
        7168,
        True,
    )


if __name__ == '__main__':
    run()
