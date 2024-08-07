# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.incubate.nn.attn_bias import (
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    PaddedSeqLenInfo,
    SeqLenInfo,
)


def all_dtypes():
    dtypes = [paddle.float32, paddle.float64]
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        dtypes.append(paddle.float16)
        prop = paddle.device.cuda.get_device_properties()
        if prop.major >= 8:
            dtypes.append(paddle.bfloat16)
    return dtypes


class TestLowerTriangularMask(unittest.TestCase):
    @paddle.no_grad()
    def check_materialize(self, shape, dtype, has_bias=False):
        assert len(shape) >= 2
        if has_bias:
            bias = paddle.rand(shape=shape, dtype=dtype)
            mask = LowerTriangularMaskWithTensorBias(bias)
        else:
            mask = LowerTriangularMask()

        mask = mask.materialize(shape=shape, dtype=dtype)
        self.assertEqual(mask.dtype, dtype)
        self.assertEqual(mask.shape, shape)
        dst_shape = [-1, mask.shape[-2], mask.shape[-1]]

        mask = mask.reshape(dst_shape).astype(paddle.float64).numpy()
        if has_bias:
            bias = bias.reshape(dst_shape).astype(paddle.float64).numpy()

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    value = mask[i][j][k]
                    if j >= k:
                        if has_bias:
                            self.assertEqual(value, bias[i][j][k])
                        else:
                            self.assertEqual(value, 0)
                    else:
                        self.assertEqual(value, float('-inf'))

    def test_materialize(self):
        shape = [5, 6, 7]
        for dtype in all_dtypes():
            for has_bias in [False, True]:
                self.check_materialize(shape, dtype, has_bias)


def check_split_tensor_without_batch_sizes(seqinfo, extra_shape):
    seqlens = []
    for i in range(len(seqinfo.seqstart_py) - 1):
        seqlens.append(seqinfo.seqstart_py[i + 1] - seqinfo.seqstart_py[i])
    shape = [1, seqinfo.seqstart_py[-1], *extra_shape]

    x = paddle.rand(shape)
    tensors = seqinfo.split(x)
    for i, t in enumerate(tensors):
        assert t.shape[0] == 1
        assert t.shape[1] == seqlens[i]
        assert t.shape[2:] == x.shape[2:]
    concated_x = paddle.concat(tensors, axis=1)
    np.testing.assert_equal(x.numpy(), concated_x.numpy())
    return x, tensors


def check_split_tensor_with_batch_sizes(seqinfo, extra_shape, batch_sizes):
    seqlens = []
    for i in range(len(seqinfo.seqstart_py) - 1):
        seqlens.append(seqinfo.seqstart_py[i + 1] - seqinfo.seqstart_py[i])

    cumsum_bs = 0
    uniq_seqlens = []
    for bs in batch_sizes:
        start = cumsum_bs
        end = cumsum_bs + bs
        for s in seqlens[start:end]:
            assert s == seqlens[start]
        cumsum_bs += bs
        uniq_seqlens.append(seqlens[start])

    x = paddle.rand(shape=[1, sum(seqlens), *extra_shape])
    tensors = seqinfo.split(x, batch_sizes)
    assert len(tensors) == len(batch_sizes)
    for i, t in enumerate(tensors):
        shape = t.shape
        assert len(shape) == 2 + len(extra_shape)
        assert shape[0] == batch_sizes[i]
        assert shape[1] == uniq_seqlens[i]

    concated_tensor = paddle.concat(
        [t.reshape([-1, *t.shape[2:]]) for t in tensors]
    ).unsqueeze(0)
    np.testing.assert_equal(x.numpy(), concated_tensor.numpy())
    return x, tensors


def check_split_tensor(seqinfo, extra_shape, batch_sizes):
    if batch_sizes is None:
        return check_split_tensor_without_batch_sizes(seqinfo, extra_shape)
    else:
        return check_split_tensor_with_batch_sizes(
            seqinfo, extra_shape, batch_sizes
        )


def check_same_tensor_list(tensors1, tensors2):
    assert len(tensors1) == len(tensors2)
    for t1, t2 in zip(tensors1, tensors2):
        assert t1.shape == t2.shape
        assert t1.dtype == t2.dtype
        np.testing.assert_equal(t1.numpy(), t2.numpy())


class TestSeqLenInfo(unittest.TestCase):
    def test_seq_len_info(self):
        n = 100
        seqlens = np.random.randint(2, 100, size=[n]).tolist()
        cumsum_seqlens = [0, *np.cumsum(seqlens).tolist()]
        info = SeqLenInfo.from_seqlens(seqlens)
        self.assertEqual(max(seqlens), info.max_seqlen)
        np.testing.assert_equal(cumsum_seqlens, info.seqstart.numpy())
        np.testing.assert_equal(cumsum_seqlens, info.seqstart_py)
        intervals = list(info.intervals())
        self.assertEqual(n, len(intervals))
        for i in range(n):
            self.assertEqual(cumsum_seqlens[i], intervals[i][0])
            self.assertEqual(cumsum_seqlens[i + 1], intervals[i][1])

        check_split_tensor_without_batch_sizes(info, [8, 9])

    def test_split_with_batch_sizes(self):
        n_tensor = 10
        extra_shape = [3, 4]
        batch_sizes = np.random.randint(10, 200, size=[n_tensor]).tolist()
        seqlens = []
        uniq_seqlens = []
        for bs in batch_sizes:
            tmp_seqlen = np.random.randint(10, 200, size=[1])[0]
            uniq_seqlens.append(tmp_seqlen)
            seqlens.extend([tmp_seqlen] * bs)
        info = SeqLenInfo.from_seqlens(seqlens)

        check_split_tensor_with_batch_sizes(info, extra_shape, batch_sizes)


class TestPaddedSeqLenInfo(unittest.TestCase):
    def test_padded_seq_len_info(self):
        n = 100
        padding = 200
        seqlens = np.random.randint(2, padding, size=[n]).tolist()
        info = PaddedSeqLenInfo.from_seqlens_padded(seqlens, padding)
        self.assertEqual(max(seqlens), info.max_seqlen)
        np.testing.assert_equal(info.seqstart.numpy(), info.seqstart_py)
        self.assertEqual(len(info.seqstart_py), n + 1)
        self.assertEqual(info.seqstart_py[0], 0)
        self.assertTrue(np.all(np.diff(info.seqstart_py) == padding))
        intervals = list(info.intervals())
        self.assertEqual(len(intervals), n)
        for i in range(n):
            interval = intervals[i]
            self.assertEqual(interval[0], padding * i)
            self.assertEqual(interval[1] - interval[0], seqlens[i])


class TestBlockDiagonalMask(unittest.TestCase):
    def setUp(self):
        self.mask_class = BlockDiagonalMask
        self.q_n = 10
        self.qkv_same_length = True
        self.config()

    def config(self):
        pass

    def test_from_seq_lens(self):
        q_seqlen = np.random.randint(2, 100, self.q_n).tolist()
        if self.qkv_same_length:
            kv_seqlen = q_seqlen
        else:
            kv_seqlen = np.random.randint(2, 100, int(self.q_n)).tolist()

        mask = self.mask_class.from_seqlens(q_seqlen, kv_seqlen)
        self.check_main(mask, q_seqlen, kv_seqlen, [3, 4])

    def test_from_tensor_list(self):
        shapes = [[2, 3], [7, 9], [11, 5]]
        extra_shape = [13, 19]
        tensors = []
        seqlens = []
        for s in shapes:
            tmp_s = s + extra_shape
            tensors.append(paddle.rand(tmp_s))
            seqlens.extend([tmp_s[1]] * tmp_s[0])
        mask, concated_tensor = self.mask_class.from_tensor_list(tensors)
        self.check_main(mask, seqlens, seqlens, extra_shape)

    def test_from_tensor_lists_qk(self):
        self.check_from_tensor_lists_qkv()

    def test_from_tensor_lists_qkv(self):
        self.check_from_tensor_lists_qkv(has_value=True)

    def check_from_tensor_lists_qkv(self, has_value=False):
        batch_sizes = [2, 3, 4]
        q_uniq_seqlens = [5, 6, 7]
        k_uniq_seqlens = [8, 9, 10]
        extra_shape = [13, 19]

        tensors_q = []
        tensors_k = []
        tensors_v = [] if has_value else None
        q_seqlens = []
        kv_seqlens = []
        for i, bs in enumerate(batch_sizes):
            q_shape = [bs, q_uniq_seqlens[i], *extra_shape]
            kv_shape = [bs, k_uniq_seqlens[i], *extra_shape]
            tensors_q.append(paddle.rand(q_shape))
            tensors_k.append(paddle.rand(kv_shape))
            q_seqlens.extend([q_shape[1]] * q_shape[0])
            kv_seqlens.extend([kv_shape[1]] * kv_shape[0])
            if has_value:
                tensors_v.append(paddle.rand(kv_shape))

        mask, q, k, v = self.mask_class.from_tensor_lists_qkv(
            tensors_q, tensors_k, tensors_v
        )
        self.check_main(
            mask,
            q_seqlens,
            kv_seqlens,
            extra_shape,
            check_same_shape_split=False,
        )

    def check_main(
        self,
        mask,
        q_seqlen,
        kv_seqlen,
        extra_shape,
        check_same_shape_split=True,
    ):
        total_q_tokens = sum(q_seqlen)
        total_kv_tokens = sum(kv_seqlen)
        shape = [*extra_shape, total_q_tokens, total_kv_tokens]
        mask_value = mask.materialize(shape=shape)
        self.assertEqual(mask_value.shape, shape)

        mask_value = mask_value.numpy()
        mask_value = mask_value.reshape([-1, *mask_value.shape[-2:]])
        for i in range(1, mask_value.shape[0]):
            np.testing.assert_equal(mask_value[i], mask_value[0])
        mask_value = mask_value[0]
        self.check_mask(
            mask_value,
            list(mask.q_seqinfo.intervals()),
            list(mask.k_seqinfo.intervals()),
        )

        x, tensors = check_split_tensor(
            mask.q_seqinfo, extra_shape, mask._batch_sizes
        )
        check_same_tensor_list(mask.split_queries(x), tensors)

        x, tensors = check_split_tensor(
            mask.k_seqinfo, extra_shape, mask._batch_sizes
        )
        check_same_tensor_list(mask.split_kv(x), tensors)

        if self.qkv_same_length and check_same_shape_split:
            x, tensors = check_split_tensor(
                mask.q_seqinfo, extra_shape, mask._batch_sizes
            )
            check_same_tensor_list(mask.split(x), tensors)

        if self.mask_class == BlockDiagonalMask:
            self.assertEqual(type(mask.make_causal()), BlockDiagonalCausalMask)

    def check_mask(self, mask, q_intervals, k_intervals):
        self.assertEqual(len(mask.shape), 2)
        m, n = mask.shape
        self.assertEqual(len(q_intervals), len(k_intervals))
        for (q_start, q_end), (k_start, k_end) in zip(q_intervals, k_intervals):
            if k_start > 0:
                self.assertTrue(
                    np.all(mask[q_start:q_end, 0:k_start] == float('-inf'))
                )
            if k_end < n:
                self.assertTrue(
                    np.all(mask[q_start:q_end, k_end:] == float('-inf'))
                )
            block_mask = mask[q_start:q_end, k_start:k_end]
            self.check_block_mask(block_mask)

    def check_block_mask(self, block_mask):
        self.assertTrue(np.all(block_mask == 0))


class TestBlockDiagonalMaskQKVDiffLength(TestBlockDiagonalMask):
    def config(self):
        self.qkv_same_length = False


class TestBlockDiagonalCausalMask(TestBlockDiagonalMask):
    def config(self):
        self.mask_class = BlockDiagonalCausalMask

    def check_block_mask(self, block_mask):
        self.assertEqual(len(block_mask.shape), 2)
        m, n = block_mask.shape
        for i in range(m):
            for j in range(n):
                if i >= j:
                    self.assertEqual(block_mask[i][j], 0)
                else:
                    self.assertEqual(block_mask[i][j], float('-inf'))


class TestBlockDiagonalCausalMaskQKVDiffLength(TestBlockDiagonalCausalMask):
    def config(self):
        self.mask_class = BlockDiagonalCausalMask
        self.qkv_same_length = False


class TestBlockDiagonalCausalWithOffsetPaddedKeysMask(unittest.TestCase):
    def test_main(self):
        kv_padding = 20
        n = 10
        extra_shape = [3, 4]

        q_seqlen = np.random.randint(0, kv_padding, size=[n]).tolist()
        kv_seqlen = np.random.randint(0, kv_padding, size=[n]).tolist()

        q_ntokens = sum(q_seqlen)
        kv_ntokens = n * kv_padding
        max_causal_diagonal = min(q_ntokens, kv_ntokens) - 2
        causal_diagonal_np = np.random.randint(
            0, max_causal_diagonal, size=[n]
        ).astype(np.int32)
        causal_diagonal = paddle.to_tensor(causal_diagonal_np)

        mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen, kv_padding, kv_seqlen, causal_diagonal
        )

        shape = [*extra_shape, q_ntokens, kv_ntokens]
        mask_np = mask.materialize(shape).numpy()
        self.assertEqual(list(mask_np.shape[: len(extra_shape)]), extra_shape)
        mask_np = mask_np.reshape([-1, *mask_np.shape[2:]])
        for i in range(1, mask_np.shape[0]):
            np.testing.assert_equal(mask_np[i], mask_np[0])

        mask_np = mask_np[0]
        q_intervals = list(mask.q_seqinfo.intervals())
        k_intervals = list(mask.k_seqinfo.intervals())
        self.assertEqual(len(q_intervals), len(k_intervals))
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(q_intervals, k_intervals)
        ):
            if k_start != 0:
                np.testing.assert_equal(
                    mask_np[q_start:q_end, 0:k_start], float('-inf')
                )

            np.testing.assert_equal(
                mask_np[q_start:q_end, k_start:k_end],
                self.create_numpy_block_mask(
                    (q_end - q_start, k_end - k_start), causal_diagonal_np[i]
                ),
            )
            if k_end != kv_ntokens:
                np.testing.assert_equal(
                    mask_np[q_start:q_end, k_end:kv_ntokens], float('-inf')
                )

    def create_numpy_block_mask(self, shape, offset, dtype=np.float32):
        t = np.full(shape, dtype=dtype, fill_value=float('-inf'))
        return np.triu(t, 1 + offset)


if __name__ == "__main__":
    unittest.main()
