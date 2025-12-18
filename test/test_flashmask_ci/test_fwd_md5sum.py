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

import itertools
import json
import os
from functools import partial

import numpy as np
import pytest
from generate_startend_row_indices import (
    generate_causal_blockwise_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_global_sliding_window_mask,
    generate_prefix_lm_causal_mask,
    generate_prefix_lm_document_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask,
    generate_share_question_mask,
    generate_sliding_window_mask,
)

import paddle
from paddle.nn.functional.flash_attention import flashmask_attention

GEN_FUNCTIONS = [
    # partial(generate_none_mask, causal=False),
    # partial(generate_none_mask, causal=True),
    partial(generate_sliding_window_mask),
    partial(generate_causal_document_mask),
    partial(generate_document_mask),
    partial(generate_share_question_mask),
    partial(generate_global_sliding_window_mask),
    partial(generate_causal_blockwise_mask),
    partial(generate_prefix_lm_document_mask),
    partial(generate_prefix_lm_causal_mask),
    partial(generate_qk_sparse_mask),
    partial(generate_random_eviction_mask),
]


def record_gt(output_file="flashmask_gt.json"):
    gt_records = {}

    param_combinations = generate_all_param_combinations()

    print(
        f"Start recording test cases, {len(param_combinations)} test cases in total."
    )

    for i, params in enumerate(param_combinations):
        try:
            out = run_flashmask_forward(**params)
            md5sum = out._md5sum()
            param_key = generate_param_key(params)

            gt_records[param_key] = md5sum
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(param_combinations)} test cases recorded")

        except Exception as e:
            print(f"Skipping test case due to exception: {params}: {e}")
            continue
    gt_records["gt_commit_id"] = input(
        "Please input the commit ID of fwd GT md5sum: "
    )
    gt_records["gt_commit_msg"] = input(
        "Please input the commit msg of fwd GT md5sum: "
    )
    with open(output_file, 'w') as f:
        json.dump(gt_records, f, indent=2)

    print(
        f"Ground truth saved to '{output_file}', {len(gt_records)} test cases recorded."
    )
    return gt_records


def run_flashmask_forward(
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
    np.random.seed(2024)
    assert nheads % nheads_kv == 0

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    startend_row_indices, causal = gen_startend_row_indices(
        batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
    )

    if startend_row_indices is None and causal and d == 80:
        pytest.skip(
            "Skipping because running headdim 80 with flash_attn in causal mask"
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

    return out


# Shape Combination
shape_cases = [
    (1, 8192, 32768 + 1024, 2, 1),
    (2840, 32, 32, 16, 4),
    (1, 300, 300, 16, 16),
    (1, 128, 127, 1, 1),
    (2, 16384, 16383, 4, 1),
]


def generate_shapes():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
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


def generate_all_param_combinations():
    dtypes = [paddle.bfloat16]
    fa_versions = [3]
    d_dv_combinations = [(128, 128), (80, 80), (64, 64)]

    shapes = list(generate_shapes())
    gen_funcs = GEN_FUNCTIONS

    combinations = []
    for (
        batch_size,
        seqlen_q,
        seqlen_k,
        nheads,
        nheads_kv,
        nheads_startend_row_indices,
    ), dtype, fa_version, (d, dv), gen_func in itertools.product(
        shapes, dtypes, fa_versions, d_dv_combinations, gen_funcs
    ):
        params = {
            'batch_size': batch_size,
            'seqlen_q': seqlen_q,
            'seqlen_k': seqlen_k,
            'nheads': nheads,
            'nheads_kv': nheads_kv,
            'd': d,
            'dv': dv,
            'nheads_startend_row_indices': nheads_startend_row_indices,
            'fa_version': fa_version,
            'dtype': dtype,
            'gen_startend_row_indices': gen_func,
            'softcap': 0.0,
        }
        combinations.append(params)
    return combinations


def generate_param_key(params):
    gen_func_index = get_gen_func_index(params['gen_startend_row_indices'])
    nheads_startend = params['nheads_startend_row_indices']
    dtype_index = get_dtype_index(params['dtype'])

    if isinstance(nheads_startend, (list, tuple)):
        nheads_startend_str = '_'.join(map(str, nheads_startend))
    else:
        nheads_startend_str = str(nheads_startend)

    return (
        f"gen_startend_row_indices{gen_func_index}-"
        f"{params['batch_size']}-{params['seqlen_q']}-{params['seqlen_k']}-"
        f"{params['nheads']}-{params['nheads_kv']}-{nheads_startend_str}-"
        f"{params['d']}-{params['dv']}-{params['fa_version']}-dtype{dtype_index}"
    )


def get_gen_func_index(gen_func):
    for i, func in enumerate(GEN_FUNCTIONS):
        if gen_func == func or (
            hasattr(gen_func, 'func') and gen_func.func == func.func
        ):
            return i
    return -1


def get_dtype_index(dtype):
    dtype_list = [paddle.bfloat16]
    for i, dt in enumerate(dtype_list):
        if dtype == dt:
            return i
    return -1


gt_records = {}
try:
    with open("flashmask_gt.json", 'r') as f:
        gt_records = json.load(f)
except FileNotFoundError:
    pass


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
@pytest.mark.parametrize("fa_version", [3])
@pytest.mark.parametrize("d, dv", [(128, 128), (80, 80), (64, 64)])
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices",
    list(generate_shapes()),
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    GEN_FUNCTIONS,
)
def test_flashmask_md5(
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
    params = {
        'batch_size': batch_size,
        'seqlen_q': seqlen_q,
        'seqlen_k': seqlen_k,
        'nheads': nheads,
        'nheads_kv': nheads_kv,
        'd': d,
        'dv': dv,
        'nheads_startend_row_indices': nheads_startend_row_indices,
        'fa_version': fa_version,
        'dtype': dtype,
        'gen_startend_row_indices': gen_startend_row_indices,
        'softcap': softcap,
    }

    param_key = generate_param_key(params)

    if param_key not in gt_records:
        pytest.skip(f"No ground truth record for {param_key}")

    out = run_flashmask_forward(**params)

    actual_md5 = out._md5sum()
    expected_md5 = gt_records[param_key]

    assert actual_md5 == expected_md5, (
        f"MD5 mismatch for {param_key}\nExpected: {expected_md5}\nGot: {actual_md5}"
    )


if __name__ == "__main__":
    if not os.path.exists("flashmask_gt.json"):
        print("Start recording ground truth...")
        record_gt()
    else:
        print("Ground truth file exists, run pytest to execute tests")
