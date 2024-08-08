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
    if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
        return []

    dtypes = [
        paddle.float32,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]
    if 'A100' in paddle.device.cuda.get_device_properties().name:
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


if __name__ == "__main__":
    unittest.main()
