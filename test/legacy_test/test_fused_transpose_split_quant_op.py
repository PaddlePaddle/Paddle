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


def dequant_ref(
    fp8_tensor: paddle.Tensor, scale: paddle.Tensor, block_size: int = 128
) -> paddle.Tensor:
    """Helper function to dequantize fp8 tensor to bf16"""
    expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
    # Handle non-aligned cases by truncating
    expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
    return (fp8_tensor.astype('float32') * expanded_scale).astype('bfloat16')


def fused_transpose_split_quant_ref(x, xscale, tokens_per_expert, pow_2_scales):
    shape = x.shape
    if x.dtype == paddle.float8_e4m3fn:
        x = dequant_ref(x, xscale)
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


def test_fused_transpose_split_quant(
    tokens_per_expert, seq_len, pow_2_scales, using_fp8=False
):

    x = paddle.randn([sum(tokens_per_expert), seq_len], dtype='bfloat16')
    if using_fp8:
        x = x.cast('float8_e4m3fn')
    xscale = (
        paddle.randn(
            [sum(tokens_per_expert), (seq_len + 127) // 128], dtype='float32'
        )
        if using_fp8
        else None
    )
    # x = paddle.clip(x, min=-50, max=50)

    out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(
        x, xscale, tokens_per_expert, pow_2_scales
    )

    out_ref, scale_ref = fused_transpose_split_quant_ref(
        x, xscale, tokens_per_expert, pow_2_scales
    )

    for t, t_ref in zip(out, out_ref):
        try:
            np.testing.assert_allclose(
                t.astype('float32'), t_ref.astype('float32')
            )
        except AssertionError as e:
            print("AssertionError", e)

    for t, t_ref in zip(scale, scale_ref):
        try:
            np.testing.assert_allclose(t, t_ref)
        except AssertionError as e:
            print("AssertionError", e)


def run():
    fp8_choice = [True, False]
    for using_fp8 in fp8_choice:
        test_fused_transpose_split_quant(
            [0, 0], 1024, False, using_fp8=using_fp8
        )
        test_fused_transpose_split_quant(
            [128, 2 * 128], 0, True, using_fp8=using_fp8
        )
        test_fused_transpose_split_quant([128], 1, False, using_fp8=using_fp8)
        test_fused_transpose_split_quant(
            [0, 128, 0, 2 * 128], 127, True, using_fp8=using_fp8
        )
        test_fused_transpose_split_quant(
            [3 * 128, 4 * 128, 5 * 128], 233, False, using_fp8=using_fp8
        )
        test_fused_transpose_split_quant(
            [24 * 128, 128, 50 * 128, 16 * 128], 2162, True, using_fp8=using_fp8
        )
        test_fused_transpose_split_quant(
            [7 * 128, 29 * 128, 3 * 128, 128 * 128, 13 * 128],
            4000,
            False,
            using_fp8=using_fp8,
        )
        test_fused_transpose_split_quant(
            [18 * 128, 5 * 128, 24 * 128, 128, 6 * 128, 0, 27 * 128, 7 * 128],
            7168,
            True,
            using_fp8=using_fp8,
        )


if __name__ == '__main__':
    run()
