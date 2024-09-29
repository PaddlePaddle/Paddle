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

import unittest

import numpy as np

import paddle


class TestPcaLowrankAPI(unittest.TestCase):
    def transpose(self, x):
        shape = x.shape
        perm = list(range(0, len(shape)))
        perm = perm[:-2] + [perm[-1]] + [perm[-2]]
        return paddle.transpose(x, perm)

    def random_matrix(self, rows, columns, *batch_dims, **kwargs):
        dtype = kwargs.get('dtype', paddle.float64)

        x = paddle.randn((*batch_dims, rows, columns), dtype=dtype)
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

    def run_subtest(
        self, guess_rank, actual_rank, matrix_size, batches, pca, **options
    ):
        if isinstance(matrix_size, int):
            rows = columns = matrix_size
        else:
            rows, columns = matrix_size
        a_input = self.random_lowrank_matrix(
            actual_rank, rows, columns, *batches
        )
        a = a_input

        u, s, v = pca(a_input, q=guess_rank, **options)

        self.assertEqual(s.shape[-1], guess_rank)
        self.assertEqual(u.shape[-2], rows)
        self.assertEqual(u.shape[-1], guess_rank)
        self.assertEqual(v.shape[-1], guess_rank)
        self.assertEqual(v.shape[-2], columns)

        A1 = u.matmul(paddle.diag_embed(s)).matmul(self.transpose(v))
        ones_m1 = paddle.ones((*batches, rows, 1), dtype=a.dtype)
        c = a.sum(axis=-2) / rows
        c = c.reshape((*batches, 1, columns))
        A2 = a - ones_m1.matmul(c)
        np.testing.assert_allclose(A1.numpy(), A2.numpy(), atol=1e-5)

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
        for actual_rank, size in [
            (2, (17, 4)),
            (2, (100, 4)),
            (6, (100, 40)),
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


class TestStaticPcaLowrankAPI(unittest.TestCase):
    def transpose(self, x):
        shape = x.shape
        perm = list(range(0, len(shape)))
        perm = perm[:-2] + [perm[-1]] + [perm[-2]]
        return paddle.transpose(x, perm)

    def random_matrix(self, rows, columns, *batch_dims, **kwargs):
        dtype = kwargs.get('dtype', 'float64')

        x = paddle.randn((*batch_dims, rows, columns), dtype=dtype)
        u, _, vh = paddle.linalg.svd(x, full_matrices=False)
        k = min(rows, columns)
        s = paddle.linspace(1 / (k + 1), 1, k, dtype=dtype)
        return (u * s.unsqueeze(-2)) @ vh

    def random_lowrank_matrix(self, rank, rows, columns, *batch_dims, **kwargs):
        B = self.random_matrix(rows, rank, *batch_dims, **kwargs)
        C = self.random_matrix(rank, columns, *batch_dims, **kwargs)
        return B.matmul(C)

    def run_subtest(
        self, guess_rank, actual_rank, matrix_size, batches, pca, **options
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            a_input = self.random_lowrank_matrix(
                actual_rank, rows, columns, *batches
            )
            a = a_input

            u, s, v = pca(a_input, q=guess_rank, **options)

            self.assertEqual(s.shape[-1], guess_rank)
            self.assertEqual(u.shape[-2], rows)
            self.assertEqual(u.shape[-1], guess_rank)
            self.assertEqual(v.shape[-1], guess_rank)
            self.assertEqual(v.shape[-2], columns)

            A1 = u.matmul(paddle.diag_embed(s)).matmul(self.transpose(v))
            ones_m1 = paddle.ones((*batches, rows, 1), dtype=a.dtype)
            c = a.sum(axis=-2) / rows
            c = c.reshape((*batches, 1, columns))
            A2 = a - ones_m1.matmul(c)
            detect_rank = (s.abs() > 1e-5).sum(axis=-1)
            left1 = actual_rank * paddle.ones(batches, dtype=paddle.int64)
            S = paddle.linalg.svd(A2, full_matrices=False)[1]
            left2 = s[..., :actual_rank]
            right = S[..., :actual_rank]

            exe = paddle.static.Executor()
            exe.run(startup)
            A1, A2, left1, detect_rank, left2, right = exe.run(
                main,
                feed={},
                fetch_list=[A1, A2, left1, detect_rank, left2, right],
            )

            np.testing.assert_allclose(A1, A2, atol=1e-5)
            if not left1.shape:
                np.testing.assert_allclose(int(left1), int(detect_rank))
            else:
                np.testing.assert_allclose(left1, detect_rank)
            np.testing.assert_allclose(left2, right)

    def test_forward(self):
        with paddle.pir_utils.IrGuard():
            pca_lowrank = paddle.linalg.pca_lowrank
            all_batches = [(), (1,), (3,), (2, 3)]
            for actual_rank, size in [
                (2, (17, 4)),
                (2, (100, 4)),
                (6, (100, 40)),
            ]:
                for batches in all_batches:
                    for guess_rank in [
                        actual_rank,
                        actual_rank + 2,
                        actual_rank + 6,
                    ]:
                        if guess_rank <= min(*size):
                            self.run_subtest(
                                guess_rank,
                                actual_rank,
                                size,
                                batches,
                                pca_lowrank,
                            )
                            self.run_subtest(
                                guess_rank,
                                actual_rank,
                                size[::-1],
                                batches,
                                pca_lowrank,
                            )


if __name__ == "__main__":
    unittest.main()
