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

from functools import partial

import pytest
from generate_startend_row_indices import (
    generate_causal_blockwise_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_global_sliding_window_mask,
    generate_none_mask,
    generate_prefix_lm_causal_mask,
    generate_prefix_lm_document_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask,
    generate_share_question_mask,
    generate_sliding_window_mask,
    startend_row_indices_to_attn_bias,
)
from test_util import attention_ref

import paddle
from paddle.nn.functional.flash_attention import flashmask_attention

# batch_size, seqlen_q, seqlen_k, nheads, nheads_kv
shape_cases = [
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    # (2, 8192, 32768, 32, 4), # this will oom
    # (2, 8192, 8192, 32, 4), # this will oom
    (2, 8192, 8192, 14, 1),
    (2, 16384, 16384, 4, 1),
    (1, 1, 127, 1, 1),
    (1, 128, 127, 1, 1),
    (1, 127, 128, 1, 1),
    (2, 16383, 16384, 4, 1),
    (2, 16384, 16383, 4, 1),
    (2, 1000, 1000, 4, 1),
    (2, 2000, 2000, 4, 1),
    (2, 3000, 3000, 4, 1),
    (1, 4000, 4000, 1, 1),
    (1, 8192, 32768 + 1024, 2, 1),
    (1, 8192, 16384 + 1024, 2, 1),
    # my case
]


# Generate all combinations for second param
def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        if nheads_kv == 1:
            nheads_startend_row_indices_values = [1]
        else:
            nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size,
                seqlen_q,
                seqlen_k,
                nheads,
                nheads_kv,
                nheads_startend_row_indices,
            )


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", [3])
@pytest.mark.parametrize("d, dv", [(128, 128), (80, 80), (64, 64), (256, 256)])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes()),
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        partial(generate_none_mask, causal=False),  # full
        partial(generate_none_mask, causal=True),  # causal
        partial(generate_sliding_window_mask),  # sliding window
        partial(generate_causal_document_mask),  # causal document mask
        partial(generate_document_mask),  # document mask
        partial(generate_share_question_mask),  # share question mask
        partial(generate_global_sliding_window_mask),  # global sliding window
        partial(generate_causal_blockwise_mask),  # causal blockwise mask
        partial(generate_prefix_lm_document_mask),  # prefix lm document mask
        partial(generate_prefix_lm_causal_mask),  # prefix lm causal mask
        partial(generate_qk_sparse_mask),  # qk-sparse mask
        partial(generate_random_eviction_mask),  # random eviction mask
    ],
)
def test_flashmask(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_kv,
    d,
    dv,
    nheads_startend_row_indices,
    fa_version,
    dtype,
    gen_startend_row_indices,
    softcap=0.0,
):
    paddle.seed(2024)
    assert nheads % nheads_kv == 0
    q_ref = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k_ref = paddle.randn(
        shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype
    )
    v_ref = paddle.randn(
        shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype
    )

    q_ref.stop_gradient = False
    k_ref.stop_gradient = False
    v_ref.stop_gradient = False

    q_bf16, k_bf16, v_bf16 = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q_bf16.stop_gradient = False
    k_bf16.stop_gradient = False
    v_bf16.stop_gradient = False

    q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    if startend_row_indices is None and causal and d == 80:
        pytest.skip(
            "Skipping because running headdim 80 with flash_attn in causal mask"
        )

    attn_bias = startend_row_indices_to_attn_bias(
        startend_row_indices, seqlen_q, nheads, dtype, causal
    )

    out_ref, attn_ref = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, attn_bias=attn_bias
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
    )

    # # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert softcap == 0.0
    rtol = 2 if softcap == 0.0 else 3

    print(
        f"Paddle naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}"
    )
    print(
        f"Paddle naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}"
    )

    if fa_version == 2:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fa_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.

    assert (out - out_ref).abs().max().item() <= rtol * (
        out_bf16 - out_ref
    ).abs().max().item() + fwd_atol

    g = paddle.randn(shape=out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    print(f"flashmask dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"flashmask dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"flashmask dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(
        f"flashmask dQ mean diff: {(q.grad - q_ref.grad).abs().mean().item()}"
    )
    print(
        f"flashmask dK mean diff: {(k.grad - k_ref.grad).abs().mean().item()}"
    )
    print(
        f"flashmask dV mean diff: {(v.grad - v_ref.grad).abs().mean().item()}"
    )

    print(
        f"Paddle naive bf16 dQ max diff: {(q_bf16.grad - q_ref.grad).abs().max().item()}"
    )
    print(
        f"Paddle naive bf16 dK max diff: {(k_bf16.grad - k_ref.grad).abs().max().item()}"
    )
    print(
        f"Paddle naive bf16 dV max diff: {(v_bf16.grad - v_ref.grad).abs().max().item()}"
    )
    print(
        f"Paddle naive bf16 dQ mean diff: {(q_bf16.grad - q_ref.grad).abs().mean().item()}"
    )
    print(
        f"Paddle naive bf16 dK mean diff: {(k_bf16.grad - k_ref.grad).abs().mean().item()}"
    )
    print(
        f"Paddle naive bf16 dV mean diff: {(v_bf16.grad - v_ref.grad).abs().mean().item()}"
    )

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (
        0 if softcap == 0 else 3e-4
    )
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (
        q_bf16.grad - q_ref.grad
    ).abs().max().item() + dq_atol
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (
        0 if softcap == 0 else 3e-4
    )
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (
        k_bf16.grad - k_ref.grad
    ).abs().max().item() + dk_atol
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (
        0 if softcap == 0 else 3e-4
    )
    assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (
        v_bf16.grad - v_ref.grad
    ).abs().max().item() + dv_atol
