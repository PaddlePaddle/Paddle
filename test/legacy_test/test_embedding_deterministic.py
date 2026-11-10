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
import contextlib
import random
import sys
import unittest
from itertools import product

import numpy as np
from op_test import is_custom_device

import paddle
from paddle.distributed.fleet.layers.mpu.mp_ops import _c_lookup_table


@contextlib.contextmanager
def deterministic_guard(value):
    flag_name = 'FLAGS_embedding_deterministic'
    old_value = paddle.get_flags(flag_name)[flag_name]
    paddle.set_flags({flag_name: value})
    assert paddle.get_flags(flag_name)[flag_name] == value
    yield
    paddle.set_flags({flag_name: old_value})
    assert paddle.get_flags(flag_name)[flag_name] == old_value


def to_numpy(tensor):
    if tensor.dtype in [paddle.float16, paddle.bfloat16]:
        tensor = tensor.astype(paddle.float32)
    return tensor.numpy()


def clone_weight(weight):
    if weight.dtype == paddle.bfloat16:
        weight = weight.astype(paddle.float32).numpy()
        weight = paddle.to_tensor(weight, dtype=paddle.float32).astype(
            paddle.bfloat16
        )
    else:
        weight = paddle.to_tensor(weight.numpy())
    weight.stop_gradient = False
    return weight


def embedding(ids, weight, out_grad, deterministic_level=0, rank=None):
    weight = clone_weight(weight)
    with deterministic_guard(deterministic_level):
        if rank is not None:
            vocab_size, _ = weight.shape
            start_idx = vocab_size * rank
            out = _c_lookup_table(weight, ids, start_index=start_idx)
        else:
            out = paddle.nn.functional.embedding(ids, weight)
        out.backward(out_grad.clone())
        return to_numpy(out), to_numpy(weight.grad)


def embedding_ground_truth(ids, weight, out_grad, rank=None):
    weight = clone_weight(weight.astype(paddle.float32))
    out_grad = out_grad.astype(paddle.float32)
    return embedding(ids, weight, out_grad, deterministic_level=2, rank=rank)


def generate_input_data(
    ids_shape,
    vocab_size,
    hidden_size,
    weight_dtype,
    ids_dtype,
    allow_duplicate_id=True,
    rank=None,
    nranks=None,
    allow_pure_random=False,
):
    max_id = vocab_size if rank is None else vocab_size * nranks
    if allow_duplicate_id:
        ids = np.random.randint(low=0, high=max_id, size=ids_shape)
    else:
        sequence = list(range(max_id))
        numel = int(np.prod(ids_shape))
        if len(sequence) < numel:
            return None, None, None
        ids = np.array(random.sample(sequence, numel)).reshape(ids_shape)

    ids = paddle.to_tensor(ids).astype(ids_dtype)
    ids.stop_gradient = True

    weight = paddle.randn([vocab_size, hidden_size]).astype(weight_dtype)
    weight.stop_gradient = False

    out_grad_shape = [*ids_shape, hidden_size]
    if allow_duplicate_id and not allow_pure_random:
        out_grad = paddle.randint(low=-10, high=10, shape=out_grad_shape)
    else:
        out_grad = paddle.randn(out_grad_shape)
    out_grad = out_grad.astype(weight.dtype)
    return ids, weight, out_grad


def get_all_dtypes():
    if (
        not (paddle.is_compiled_with_cuda() or is_custom_device())
        or paddle.is_compiled_with_rocm()
    ):
        return []

    dtypes = [
        paddle.float32,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]
    if 'A100' in paddle.device.get_device_properties().name:
        dtypes.append(paddle.bfloat16)
    return dtypes


class TestEmbeddingBase(unittest.TestCase):
    def setUp(self):
        self.ids_shape = [32, 3]
        self.vocab_size = 128
        self.hidden_size = 1024
        self.nranks = 8

    def check_main(
        self,
        weight_dtype,
        ids_dtype,
        deterministic_level=0,
        rank=None,
        allow_duplicate_id=True,
        allow_pure_random=False,
    ):
        if sys.platform == 'win32' and rank is not None:
            return

        ids, weight, out_grad = generate_input_data(
            ids_shape=self.ids_shape,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            weight_dtype=weight_dtype,
            ids_dtype=ids_dtype,
            allow_duplicate_id=allow_duplicate_id,
            rank=rank,
            nranks=self.nranks,
            allow_pure_random=allow_pure_random,
        )
        if ids is None:
            return

        if allow_pure_random:
            out_1, weight_grad_1 = embedding_ground_truth(
                ids, weight, out_grad, rank
            )
            out_2, weight_grad_2 = embedding_ground_truth(
                ids, weight, out_grad, rank
            )
        else:
            out_1, weight_grad_1 = embedding_ground_truth(
                ids, weight, out_grad, rank
            )
            out_2, weight_grad_2 = embedding(
                ids,
                weight,
                out_grad,
                deterministic_level=deterministic_level,
                rank=rank,
            )
        np.testing.assert_equal(out_1, out_2)
        np.testing.assert_equal(weight_grad_1, weight_grad_2)

    def test_main(self):
        weight_dtypes = get_all_dtypes()
        ids_dtypes = [paddle.int64, paddle.int32]
        deterministic_levels = [0, 1]
        ranks = [None, 0, 2, 4, 8]
        allow_duplicate_ids = [False, True]
        allow_pure_randoms = [False, True]
        for (
            weight_dtype,
            ids_dtype,
            deterministic_level,
            rank,
            allow_duplicate_id,
            allow_pure_random,
        ) in product(
            weight_dtypes,
            ids_dtypes,
            deterministic_levels,
            ranks,
            allow_duplicate_ids,
            allow_pure_randoms,
        ):
            self.check_main(
                weight_dtype,
                ids_dtype,
                deterministic_level,
                rank,
                allow_duplicate_id,
                allow_pure_random,
            )


class TestEmbedding2(TestEmbeddingBase):
    def setUp(self):
        self.ids_shape = [32, 16]
        self.vocab_size = 128
        self.hidden_size = 1024
        self.nranks = 8


class TestEmbeddingDeterministic(unittest.TestCase):
    def setUp(self):
        self.ids_shape = [32, 16]
        self.vocab_size = 128
        self.hidden_size = 1024


# Keep in sync with kSortThreshold in embedding_grad_kernel.cu and
# kEmbeddingGradRowsPerPartial in embedding_grad.h.
SORT_THRESHOLD = 3072
ROWS_PER_PARTIAL = 10
FEATURE_KERNEL_CHUNK = 32


def round_bf16(x):
    # Round float32 to the nearest bfloat16 (ties to even), kept in float32.
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & (
        np.uint32(0xFFFF0000)
    )
    return u.view(np.float32)


def get_mt_and_round_fn(dtype):
    # Returns the accumulation type MT (numpy) of dtype and a function that
    # rounds an MT array to dtype while keeping it in MT.
    if dtype == paddle.float16:
        return np.float32, lambda x: x.astype(np.float16).astype(np.float32)
    if dtype == paddle.bfloat16:
        return np.float32, round_bf16
    mt = {
        paddle.float32: np.float32,
        paddle.float64: np.float64,
        paddle.complex64: np.complex64,
        paddle.complex128: np.complex128,
    }[dtype]
    return mt, lambda x: x


def feature_kernel_reference(ids, grad, n, dtype, round_acc_first):
    # Emulates EmbeddingGradDeterministicKernel: rows are processed in chunks
    # of 32, rows sharing one id in a chunk are summed in MT by row order, and
    # the chunk sum is then added to the table.
    mt, round_t = get_mt_and_round_fn(dtype)
    table = np.zeros([n, grad.shape[1]], dtype=mt)
    for start in range(0, len(ids), FEATURE_KERNEL_CHUNK):
        chunk = ids[start : start + FEATURE_KERNEL_CHUNK]
        for cur_id in dict.fromkeys(chunk.tolist()):
            rows = start + np.nonzero(chunk == cur_id)[0]
            acc = grad[rows[0]].copy()
            for row in rows[1:]:
                acc += grad[row]
            if round_acc_first:
                acc = round_t(acc)
            table[cur_id] = round_t(table[cur_id] + acc)
    return table


def sort_kernel_reference(ids, grad, n, dtype):
    # Emulates LaunchEmbeddingGradSortKernel: rows of one id are taken in
    # stable sorted order, summed in MT every ROWS_PER_PARTIAL rows, and the
    # partials are summed in MT and rounded once.
    mt, round_t = get_mt_and_round_fn(dtype)
    table = np.zeros([n, grad.shape[1]], dtype=mt)
    order = np.argsort(ids, kind='stable')
    sorted_ids = ids[order]
    for cur_id in np.unique(sorted_ids):
        rows = order[sorted_ids == cur_id]
        total = np.zeros([grad.shape[1]], dtype=mt)
        for start in range(0, len(rows), ROWS_PER_PARTIAL):
            partial = np.zeros([grad.shape[1]], dtype=mt)
            for row in rows[start : start + ROWS_PER_PARTIAL]:
                partial += grad[row]
            total += partial
        table[cur_id] = round_t(total)
    return table


@contextlib.contextmanager
def flags_guard(flags):
    old_flags = paddle.get_flags(list(flags.keys()))
    paddle.set_flags(flags)
    try:
        yield
    finally:
        paddle.set_flags(old_flags)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
    "The sort based embedding backward is only available on CUDA.",
)
class TestEmbeddingDeterministicAcrossSortThreshold(unittest.TestCase):
    # Checks FLAGS_embedding_deterministic=1 bitwise against a numpy emulation
    # of the expected accumulation order on both sides of SORT_THRESHOLD.
    def setUp(self):
        # (ids_shape, vocab_size, hidden_size)
        self.cases = [
            ([SORT_THRESHOLD], 64, 100),
            ([SORT_THRESHOLD + 1], 64, 100),
            ([SORT_THRESHOLD + 1], 8192, 64),
            ([128, 128], 1000, 130),
        ]
        self.dtypes = [
            paddle.float32,
            paddle.float16,
            paddle.complex64,
            paddle.complex128,
        ]
        if paddle.amp.is_bfloat16_supported():
            self.dtypes.append(paddle.bfloat16)
        self.ids_dtypes = [paddle.int64, paddle.int32]
        # hot: about half of the ids are one id, so that its segment is split
        # into many partials.
        self.id_modes = ['uniform', 'hot']

    def generate(self, ids_shape, vocab_size, hidden_size, dtype, id_mode):
        numel = int(np.prod(ids_shape))
        ids = np.random.randint(0, vocab_size, size=[numel])
        if id_mode == 'hot':
            ids[np.random.rand(numel) < 0.5] = vocab_size // 2
        grad_shape = [*ids_shape, hidden_size]
        if dtype in [paddle.complex64, paddle.complex128]:
            grad = np.random.randn(*grad_shape) + 1j * np.random.randn(
                *grad_shape
            )
            out_grad = paddle.to_tensor(grad).astype(dtype)
        else:
            out_grad = paddle.randn(grad_shape).astype(dtype)
        weight = paddle.randn([vocab_size, hidden_size]).astype(dtype)
        weight.stop_gradient = False
        # Use the values actually held by out_grad, already rounded to dtype.
        grad_np = to_numpy(out_grad).reshape([numel, hidden_size])
        return ids, weight, out_grad, grad_np

    def run_embedding(self, ids, weight, out_grad, compatible):
        weight = clone_weight(weight)
        with flags_guard(
            {
                'FLAGS_embedding_deterministic': 1,
                'FLAGS_use_accuracy_compatible_kernel': compatible,
            }
        ):
            out = paddle.nn.functional.embedding(ids, weight)
            out.backward(out_grad.clone())
        return to_numpy(weight.grad)

    def test_main(self):
        for (ids_shape, vocab_size, hidden_size), dtype, id_mode in product(
            self.cases, self.dtypes, self.id_modes
        ):
            ids_np, weight, out_grad, grad_np = self.generate(
                ids_shape, vocab_size, hidden_size, dtype, id_mode
            )
            for compatible in [False, True]:
                if len(ids_np) > SORT_THRESHOLD:
                    expected = sort_kernel_reference(
                        ids_np, grad_np, vocab_size, dtype
                    )
                else:
                    expected = feature_kernel_reference(
                        ids_np, grad_np, vocab_size, dtype, compatible
                    )
                for ids_dtype in self.ids_dtypes:
                    ids = paddle.to_tensor(ids_np.reshape(ids_shape)).astype(
                        ids_dtype
                    )
                    actual = self.run_embedding(
                        ids, weight, out_grad, compatible
                    )
                    np.testing.assert_array_equal(
                        actual,
                        expected,
                        err_msg=(
                            f"ids_shape={ids_shape}, vocab_size={vocab_size}, "
                            f"hidden_size={hidden_size}, dtype={dtype}, "
                            f"id_mode={id_mode}, compatible={compatible}, "
                            f"ids_dtype={ids_dtype}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
