#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random
import re
import unittest

import numpy as np

import paddle


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


class TestPcaLowrankAPI(unittest.TestCase):
    def is_sparse(self, x):
        if isinstance(x, paddle.Tensor):
            try:
                tmp = x.indices()
                return True
            except:
                return False

        error_str = "expected Tensor" + f" but got {type(x)}"
        raise TypeError(error_str)

    def transpose(self, x):
        shape = x.shape
        perm = list(range(0, len(shape)))
        perm = perm[:-2] + [perm[-1]] + [perm[-2]]
        if self.is_sparse(x):
            return paddle.sparse.transpose(x, perm)
        return paddle.transpose(x, perm)

    def random_matrix(self, rows, columns, *batch_dims, **kwargs):
        dtype = kwargs.get('dtype', paddle.float64)

        x = paddle.randn(batch_dims + (rows, columns), dtype=dtype)
        if x.numel() == 0:
            return x
        u, _, vh = paddle.linalg.svd(x, full_matrices=False)
        k = min(rows, columns)
        s = paddle.linspace(1 / (k + 1), 1, k, dtype=dtype)
        return (u * s.unsqueeze(-2)) @ vh

    def random_lowrank_matrix(self, rank, rows, columns, *batch_dims, **kwargs):
        B = self.random_matrix(rows, rank, *batch_dims, **kwargs)
        C = self.random_matrix(rank, columns, *batch_dims, **kwargs)
        return B.matmul(C)

    def random_sparse_matrix(self, rows, columns, density=0.01, **kwargs):
        dtype = kwargs.get('dtype', paddle.float64)

        nonzero_elements = max(
            min(rows, columns), int(rows * columns * density)
        )

        row_indices = [i % rows for i in range(nonzero_elements)]
        column_indices = [i % columns for i in range(nonzero_elements)]
        random.shuffle(column_indices)
        indices = [row_indices, column_indices]
        values = paddle.randn((nonzero_elements,), dtype=dtype)
        values *= paddle.to_tensor(
            [-float(i - j) ** 2 for i, j in zip(*indices)], dtype=dtype
        ).exp()
        indices_tensor = paddle.to_tensor(indices)
        x = paddle.sparse.sparse_coo_tensor(
            indices_tensor, values, (rows, columns)
        )
        return paddle.sparse.coalesce(x)

    def run_subtest(
        self, guess_rank, actual_rank, matrix_size, batches, pca, **options
    ):
        density = options.pop('density', 1)
        if isinstance(matrix_size, int):
            rows = columns = matrix_size
        else:
            rows, columns = matrix_size
        if density == 1:
            a_input = self.random_lowrank_matrix(
                actual_rank, rows, columns, *batches
            )
            a = a_input
        else:
            a_input = self.random_sparse_matrix(rows, columns, density)
            a = a_input.to_dense()

        u, s, v = pca(a_input, q=guess_rank, **options)

        self.assertEqual(s.shape[-1], guess_rank)
        self.assertEqual(u.shape[-2], rows)
        self.assertEqual(u.shape[-1], guess_rank)
        self.assertEqual(v.shape[-1], guess_rank)
        self.assertEqual(v.shape[-2], columns)

        A1 = u.matmul(paddle.nn.functional.diag_embed(s)).matmul(
            self.transpose(v)
        )
        ones_m1 = paddle.ones(batches + (rows, 1), dtype=a.dtype)
        c = a.sum(axis=-2) / rows
        c = c.reshape(batches + (1, columns))
        A2 = a - ones_m1.matmul(c)
        np.testing.assert_allclose(A1.numpy(), A2.numpy(), atol=1e-5)

        if density == 1:
            detect_rank = (s.abs() > 1e-5).sum(axis=-1)
            left = actual_rank * paddle.ones(batches, dtype=paddle.int64)
            if not left.shape:
                np.testing.assert_allclose(int(left), int(detect_rank))
            else:
                np.testing.assert_allclose(left.numpy(), detect_rank.numpy())
            S = paddle.linalg.svd(A2, full_matrices=False)[1]
            left = s[..., :actual_rank]
            right = S[..., :actual_rank]
            np.testing.assert_allclose(left.numpy(), right.numpy())

    def test_forward(self):
        pca_lowrank = paddle.linalg.pca_lowrank
        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
            (2, (17, 4), all_batches),
            (2, (100, 4), all_batches),
            (6, (100, 40), all_batches),
        ]:
            for batches in all_batches:
                for guess_rank in [
                    actual_rank,
                    actual_rank + 2,
                    actual_rank + 6,
                ]:
                    if guess_rank <= min(*size):
                        self.run_subtest(
                            guess_rank, actual_rank, size, batches, pca_lowrank
                        )
                        self.run_subtest(
                            guess_rank,
                            actual_rank,
                            size[::-1],
                            batches,
                            pca_lowrank,
                        )
        x = np.random.randn(5, 5).astype('float64')
        x = paddle.to_tensor(x)
        q = None
        U, S, V = pca_lowrank(x, q, center=False)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_sparse(self):
        # sparse input
        pca_lowrank = paddle.linalg.pca_lowrank
        for guess_rank, size in [
            (4, (17, 4)),
            (4, (4, 17)),
            (16, (17, 17)),
            (21, (100, 40)),
        ]:
            for density in [0.005, 0.01]:
                self.run_subtest(
                    guess_rank, None, size, (), pca_lowrank, density=density
                )

    def test_errors(self):
        pca_lowrank = paddle.linalg.pca_lowrank
        x = np.random.randn(5, 5).astype('float64')
        x = paddle.to_tensor(x)

        def test_x_not_tensor():
            U, S, V = pca_lowrank(x.numpy())

        self.assertRaises(ValueError, test_x_not_tensor)

        def test_q_range():
            q = -1
            U, S, V = pca_lowrank(x, q)

        self.assertRaises(ValueError, test_q_range)

        def test_niter_range():
            n = -1
            U, S, V = pca_lowrank(x, niter=n)

        self.assertRaises(ValueError, test_niter_range)


if __name__ == "__main__":
    unittest.main()
