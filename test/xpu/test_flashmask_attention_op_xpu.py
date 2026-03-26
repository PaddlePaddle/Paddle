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

import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flashmask_attention,
)


def naive_attention(query, key, value, mask):
    if query.dtype != paddle.float32:
        query = paddle.cast(query, 'float32')
        key = paddle.cast(key, 'float32')
        value = paddle.cast(value, 'float32')
        mask = paddle.cast(mask, 'float32')

    assert query.dtype == paddle.float32
    assert key.dtype == paddle.float32
    assert value.dtype == paddle.float32
    assert mask.dtype == paddle.float32

    scale = 1.0 / np.sqrt(query.shape[-1])
    query = paddle.transpose(query, [0, 2, 1, 3])
    key = paddle.transpose(key, [0, 2, 1, 3])
    value = paddle.transpose(value, [0, 2, 1, 3])
    product = paddle.matmul(x=query, y=key, transpose_y=True)

    product = paddle.scale(product, scale)

    if mask is not None:
        mask[mask == -np.inf] = -1e37
        product = product + mask

    weights = paddle.nn.functional.softmax(product)

    out = paddle.matmul(weights, value)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out


def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m


def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = (
        paddle.to_tensor(start_row_indices, dtype=paddle.int32)
        .reshape((1, 1, S, 1))
        .repeat_interleave(B, 0)
    )

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = (
        [seq_cusums[-2]] * seq_cusums[-2]
        + [seq_cusums[-1]] * doc_seq_lens[-1]
        + [S] * padding
    )
    end_row_indices = (
        paddle.to_tensor(end_row_indices, dtype=paddle.int32)
        .reshape((1, 1, S, 1))
        .repeat_interleave(B, 0)
    )

    startend_row_indices = paddle.concat(
        [start_row_indices, end_row_indices], axis=-1
    )

    causal = True
    return startend_row_indices, causal


def generate_causal_mask(B, L, H, D, doc_seq_lens):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= L
    assert len(doc_seq_lens) >= 3
    padding = L - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = (
        paddle.to_tensor(startend_row_indices, dtype=paddle.int32)
        .reshape((1, 1, L, 1))
        .repeat_interleave(B, 0)
    )

    causal = True
    return startend_row_indices, causal


def generate_share_question_mask(B, L, H, D, doc_seq_lens):
    seq_cusums = np.cumsum(doc_seq_lens)
    seq_cusums = np.append(seq_cusums, 128)

    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= L
    assert len(doc_seq_lens) >= 3
    padding = L - total_seq_len

    startend_row_indices = [total_seq_len] * doc_seq_lens[0]

    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
    startend_row_indices = (
        paddle.to_tensor(startend_row_indices, dtype=paddle.int32)
        .reshape((1, 1, L, 1))
        .repeat_interleave(B, 0)
    )

    causal = True
    return startend_row_indices, causal


@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.shape = (1, 128, 2, 32)
        self.seq_lod = [31, 28, 69]
        self.dropout = 0.0
        self.return_softmax = False

    def test_all(self):
        self.run_case(
            dtype="float16",
            tolerance=5e-4,
            tolerance_dv=1e-3,
            generate_mask_fn=generate_causal_mask,
        )
        self.run_case(
            dtype="bfloat16",
            tolerance=6e-3,
            tolerance_dv=1e-2,
            generate_mask_fn=generate_causal_mask,
        )
        self.run_case(
            dtype="float16",
            tolerance=5e-4,
            tolerance_dv=1e-3,
            generate_mask_fn=generate_share_question_mask,
        )
        self.run_case(
            dtype="bfloat16",
            tolerance=6e-3,
            tolerance_dv=1e-2,
            generate_mask_fn=generate_share_question_mask,
        )
        self.run_case(
            dtype="float16",
            tolerance=5e-4,
            tolerance_dv=1e-3,
            generate_mask_fn=generate_causal_blockwise_mask,
        )
        self.run_case(
            dtype="bfloat16",
            tolerance=6e-3,
            tolerance_dv=1e-2,
            generate_mask_fn=generate_causal_blockwise_mask,
        )

    def run_case(self, dtype, tolerance, tolerance_dv, generate_mask_fn):
        # test dynamic
        paddle.disable_static()
        B = self.shape[0]
        L = self.shape[1]
        H = self.shape[2]
        D = self.shape[3]
        np.random.seed(2023)
        query = np.random.uniform(-1.0, 1.0, self.shape)
        key = np.random.uniform(-1.0, 1.0, self.shape)
        value = np.random.uniform(-1.0, 1.0, self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        startend_row_indices, is_causal = generate_mask_fn(
            B, L, H, D, self.seq_lod
        )
        dense_mask = flashmask_to_densemask(
            startend_row_indices, "float32", is_causal
        )

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices,
            dropout=self.dropout,
            causal=is_causal,
        )

        out_ = naive_attention(q_, k_, v_, dense_mask)

        out.backward()
        out_.backward()

        # forward result
        float_out = paddle.cast(out, "float32")
        float_out_ = paddle.cast(out_, "float32")

        np.testing.assert_allclose(
            float_out, float_out_, rtol=tolerance, atol=tolerance
        )

        # backward shape
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(k_.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(v_.grad.shape, v.shape)

        # backward result
        float_q_grad = paddle.cast(q.grad, "float32")
        float_q_grad_ = paddle.cast(q_.grad, "float32")
        float_k_grad = paddle.cast(k.grad, "float32")
        float_k_grad_ = paddle.cast(k_.grad, "float32")
        float_v_grad = paddle.cast(v.grad, "float32")
        float_v_grad_ = paddle.cast(v_.grad, "float32")

        max_diff_q_grad = np.max(
            np.abs(float_q_grad.numpy() - float_q_grad_.numpy())
        )
        mean_diff_q_grad = np.mean(
            np.abs(float_q_grad.numpy() - float_q_grad_.numpy())
        )

        np.testing.assert_allclose(
            float_q_grad, float_q_grad_, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            float_k_grad, float_k_grad_, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            float_v_grad, float_v_grad_, rtol=tolerance_dv, atol=tolerance_dv
        )


@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestFlashMaskAttentionZeroSize(unittest.TestCase):
    """Test flashmask_attention with 0-size tensors on XPU.

    When any input tensor has a dimension of size 0, the function should
    return a zero tensor with the same shape as query without calling
    the XPU kernel to avoid invalid memory access.
    """

    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.dtype = 'float16'

    def _run_test(self, q_shape, k_shape, v_shape, startend_shape, causal=True):
        """Helper method to run a single test case."""
        paddle.disable_static()

        # Create tensors using paddle.to_tensor with place parameter
        q = paddle.to_tensor(
            np.random.randn(*q_shape).astype(self.dtype), place=self.place
        )
        k = paddle.to_tensor(
            np.random.randn(*k_shape).astype(self.dtype), place=self.place
        )
        v = paddle.to_tensor(
            np.random.randn(*v_shape).astype(self.dtype), place=self.place
        )
        startend = paddle.to_tensor(
            np.ones(startend_shape, dtype="int32"), place=self.place
        )

        result = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend,
            causal=causal,
        )

        # According to FlashAttnInferMeta:
        # - Output shape is based on q.shape
        # - head_dim uses v's head_dim
        # - If batch is 0 (q.batch=0 or k.batch=0 or v.batch=0), output batch is 0
        expected_shape = list(q.shape)
        expected_shape[3] = v_shape[3]  # head_dim from v
        if q_shape[0] == 0 or k_shape[0] == 0 or v_shape[0] == 0:
            expected_shape[0] = 0  # batch is 0

        # Verify result shape
        self.assertEqual(list(result.shape), expected_shape)
        # Verify result is all zeros
        self.assertTrue(paddle.all(result == 0).item())

    def test_query_zero_seqlen(self):
        """Test when query sequence length is 0."""
        self._run_test(
            q_shape=[1, 128, 8, 96],
            k_shape=[1, 128, 8, 96],
            v_shape=[1, 0, 8, 96],  # query seqlen = 0
            startend_shape=[1, 1, 128, 1],
        )

    def test_key_zero_seqlen(self):
        """Test when key sequence length is 0."""
        self._run_test(
            q_shape=[1, 128, 8, 96],
            k_shape=[1, 0, 8, 96],  # key seqlen = 0
            v_shape=[1, 0, 8, 96],  # value must have same seqlen as key
            startend_shape=[1, 1, 0, 1],
        )

    def test_key_zero_head_dim(self):
        """Test when key head_dim is 0."""
        # Since k.numel() = 0, the kernel returns zeros.
        # Output shape uses value's head_dim: [1, 128, 8, 96]
        paddle.disable_static()
        q = paddle.to_tensor(
            np.random.randn(1, 128, 8, 96).astype(self.dtype), place=self.place
        )
        k = paddle.to_tensor(
            np.random.randn(1, 128, 8, 0).astype(self.dtype), place=self.place
        )
        v = paddle.to_tensor(
            np.random.randn(1, 128, 8, 96).astype(self.dtype), place=self.place
        )
        startend = paddle.to_tensor(
            np.ones([1, 1, 128, 1], dtype="int32"), place=self.place
        )
        result = flashmask_attention(
            q, k, v, startend_row_indices=startend, causal=True
        )
        # Output shape uses value's head_dim: [1, 128, 8, 96]
        expected_shape = [1, 128, 8, 96]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertTrue(paddle.all(result == 0).item())

    def test_value_zero_batch(self):
        """Test when value batch size is 0."""
        self._run_test(
            q_shape=[1, 128, 8, 96],
            k_shape=[1, 128, 8, 96],
            v_shape=[0, 128, 8, 96],  # value batch = 0
            startend_shape=[1, 1, 128, 1],
        )

    def test_value_zero_seqlen(self):
        """Test when value sequence length is 0."""
        self._run_test(
            q_shape=[1, 128, 8, 96],
            k_shape=[1, 128, 8, 96],
            v_shape=[1, 0, 8, 96],  # value seqlen = 0
            startend_shape=[1, 1, 128, 1],
        )

    def test_value_zero_head_dim(self):
        """Test when value head_dim is 0."""
        # Since v.numel() = 0, kernel returns zeros.
        # Output shape uses value's head_dim: [1, 128, 8, 0]
        paddle.disable_static()
        q = paddle.to_tensor(
            np.random.randn(1, 128, 8, 96).astype(self.dtype), place=self.place
        )
        k = paddle.to_tensor(
            np.random.randn(1, 128, 8, 96).astype(self.dtype), place=self.place
        )
        v = paddle.to_tensor(
            np.random.randn(1, 128, 8, 0).astype(self.dtype), place=self.place
        )
        startend = paddle.to_tensor(
            np.ones([1, 1, 128, 1], dtype="int32"), place=self.place
        )
        result = flashmask_attention(
            q, k, v, startend_row_indices=startend, causal=True
        )
        # Output shape uses value's head_dim: [1, 128, 8, 0]
        expected_shape = [1, 128, 8, 0]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertTrue(paddle.all(result == 0).item())

    def test_all_zero_batch(self):
        """Test when all tensors have batch size 0."""
        self._run_test(
            q_shape=[0, 128, 8, 96],
            k_shape=[0, 128, 8, 96],
            v_shape=[0, 128, 8, 96],
            startend_shape=[0, 1, 128, 1],
        )


if __name__ == '__main__':
    unittest.main()
