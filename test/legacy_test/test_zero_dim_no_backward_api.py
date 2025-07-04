#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import os
import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle


# Use to test API whose zero-dim input tensors don't have grad and not need to test backward in OpTest.
class TestNoBackwardAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.shape = [
            paddle.full([], 2, 'int32'),
            paddle.full([], 3, 'int32'),
            paddle.full([], 4, 'int32'),
        ]

    def test_slice(self):
        starts = [paddle.full([], 1, 'int32'), paddle.full([], 1, 'int32')]
        ends = [paddle.full([], 3, 'int32'), paddle.full([], 3, 'int32')]
        x = paddle.rand([5, 3, 3])
        out = paddle.slice(x, [1, 2], starts, ends)
        self.assertEqual(out.shape, [5, 2, 2])

    def test_strided_slice(self):
        starts = [paddle.full([], 0, 'int32'), paddle.full([], 0, 'int32')]
        ends = [paddle.full([], 4, 'int32'), paddle.full([], 4, 'int32')]
        strides = [paddle.full([], 2, 'int32'), paddle.full([], 2, 'int32')]
        x = paddle.rand([5, 5, 5])
        out = paddle.strided_slice(x, [1, 2], starts, ends, strides)
        self.assertEqual(out.shape, [5, 2, 2])

    def test_linspace(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 5.0)
        num = paddle.full([], 5, 'int32')
        out = paddle.linspace(start, stop, num)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_logspace(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 3.0)
        num = paddle.full([], 5, 'int32')
        base = paddle.full([], 2.0)
        out = paddle.logspace(start, stop, num, base)
        self.assertEqual(out.shape, [5])

    def test_arange(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 6.0)
        step = paddle.full([], 1.0)
        out = paddle.arange(start, stop, step)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normal(self):
        mean = paddle.full([], 0.0)
        std = paddle.full([], 0.0)
        out = paddle.normal(mean, std)
        self.assertEqual(out.shape, [])

        out = paddle.normal(0.0, 1.0, [])
        self.assertEqual(out.shape, [])

        out = paddle.normal(0.0, 1.0, self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_rand(self):
        out = paddle.rand([])
        self.assertEqual(out.shape, [])

        out = paddle.rand(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_randn(self):
        out = paddle.randn([])
        self.assertEqual(out.shape, [])

        out = paddle.randn(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_randint_and_randint_like(self):
        out = paddle.randint(-10, 10, [])
        self.assertEqual(out.shape, [])

        out = paddle.randint_like(out, -10, 10)
        self.assertEqual(out.shape, [])

        out = paddle.randint(-10, 10, self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_standard_normal(self):
        out = paddle.standard_normal([])
        self.assertEqual(out.shape, [])

        out = paddle.standard_normal(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_uniform(self):
        out = paddle.uniform([])
        self.assertEqual(out.shape, [])

        out = paddle.uniform(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_empty_and_empty_like(self):
        out = paddle.empty([])
        self.assertEqual(out.shape, [])

        out = paddle.empty_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.empty(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_full_and_full_like(self):
        out = paddle.full([], 0.5)
        self.assertEqual(out.shape, [])

        out = paddle.full_like(out, 0.5)
        self.assertEqual(out.shape, [])

        out = paddle.full(self.shape, 0.5)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_ones_and_ones_like(self):
        out = paddle.ones([])
        self.assertEqual(out.shape, [])

        out = paddle.ones_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.ones(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_zeros_and_zeros_like(self):
        out = paddle.zeros([])
        self.assertEqual(out.shape, [])

        out = paddle.zeros_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.zeros(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        w0 = paddle.arange(3, 9).reshape((3, 2)).astype(paddle.float32)
        w = paddle.to_tensor(w0, stop_gradient=False)
        emb = paddle.nn.functional.embedding(
            x=ids, weight=w, sparse=True, name="embedding"
        )
        self.assertEqual(emb.shape, [2])
        res = [5.0, 6.0]
        for i in range(len(res)):
            self.assertEqual(emb.numpy()[i], res[i])

    def test_one_hot_label(self):
        label = paddle.full(shape=[], fill_value=2, dtype='int64')
        one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
        self.assertEqual(one_hot_label.shape, [4])
        self.assertEqual(one_hot_label.numpy()[2], 1)

    def test_unique_consecutive(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            x = paddle.rand([])
            y, inverse, counts = paddle.unique_consecutive(
                x,
                return_inverse=True,
                return_counts=True,
            )

            self.assertEqual(y, x)
            self.assertEqual(inverse, 0)
            self.assertEqual(counts, 1)
            self.assertEqual(y.shape, [1])
            self.assertEqual(inverse.shape, [1])
            self.assertEqual(counts.shape, [1])

    def test_unique(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            x = paddle.rand([])
            y, index, inverse, counts = paddle.unique(
                x,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )

            self.assertEqual(y, x)
            self.assertEqual(index, 0)
            self.assertEqual(inverse, 0)
            self.assertEqual(counts, 1)
            self.assertEqual(y.shape, [1])
            self.assertEqual(index.shape, [1])
            self.assertEqual(inverse.shape, [1])
            self.assertEqual(counts.shape, [1])

    def test_matrix_rank(self):
        x = paddle.eye(10)
        x.stop_gradient = False
        out = paddle.linalg.matrix_rank(x)

        self.assertEqual(out.shape, [])
        np.testing.assert_equal(out, np.array(10))

        c = paddle.ones(shape=[3, 4, 5])
        c.stop_gradient = False
        out_c = paddle.linalg.matrix_rank(c)
        self.assertEqual(out_c.shape, [3])
        np.testing.assert_equal(out_c, np.array([1, 1, 1]))

        # 2D, tol->float : OUTPUT 0D
        x_tol = paddle.eye(10)
        x_tol.stop_gradient = False
        out_tol = paddle.linalg.matrix_rank(x_tol, tol=0.1)
        self.assertEqual(out_tol.shape, [])

        # 3D, tol->float : OUTPUT 1D
        c_tol = paddle.ones(shape=[3, 4, 5])
        c_tol.stop_gradient = False
        out_c_tol = paddle.linalg.matrix_rank(c_tol, tol=0.1)
        self.assertEqual(out_c_tol.shape, [3])

        tol_2 = paddle.randn([2])
        # 2D, tol->Tensor[1,2] : OUTPUT 1D
        d = paddle.eye(10)
        out_d = paddle.linalg.matrix_rank(d, tol=tol_2)
        self.assertEqual(out_d.shape, [2])

    def test_eye_zero_dim_input(self):
        # use zero-dim tensor as inputs
        num_rows = paddle.to_tensor(5, stop_gradient=False)
        num_cols = paddle.to_tensor(4, stop_gradient=False)
        out = paddle.eye(num_rows, num_cols)

        self.assertEqual(num_cols.shape, [])
        self.assertEqual(num_rows.shape, [])
        self.assertEqual(out.shape, [5, 4])


class TestNoBackwardAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def create_dynamic_shape(self):
        return [
            paddle.full([], 2, 'int32'),
            paddle.full([], 3, 'int32'),
            paddle.full([], 4, 'int32'),
        ]

    def test_slice(self):
        starts = [paddle.full([], 1, 'int32'), paddle.full([], 1, 'int32')]
        ends = [paddle.full([], 3, 'int32'), paddle.full([], 3, 'int32')]
        x = paddle.rand([5, 3, 3])
        out = paddle.slice(x, [1, 2], starts, ends)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        self.assertEqual(res.shape, (5, 2, 2))

    @prog_scope()
    def test_strided_slice(self):
        starts = [paddle.full([], 0, 'int32'), paddle.full([], 0, 'int32')]
        ends = [paddle.full([], 4, 'int32'), paddle.full([], 4, 'int32')]
        strides = [paddle.full([], 2, 'int32'), paddle.full([], 2, 'int32')]
        x = paddle.rand([5, 5, 5])
        out = paddle.strided_slice(x, [1, 2], starts, ends, strides)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        self.assertEqual(res.shape, (5, 2, 2))

    def test_linspace(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 5.0)
        num = paddle.full([], 5, 'int32')
        out = paddle.linspace(start, stop, num)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        np.testing.assert_array_equal(res, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_arange(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 6.0)
        step = paddle.full([], 1.0)
        out = paddle.arange(start, stop, step)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        np.testing.assert_array_equal(res, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normal(self):
        mean = paddle.full([], 0.0)
        std = paddle.full([], 0.0)
        out1 = paddle.normal(mean, std)
        out2 = paddle.normal(0.0, 1.0, [])
        out3 = paddle.normal(0.0, 1.0, self.create_dynamic_shape())

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_rand(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.rand([])
            out2 = paddle.rand(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2, 3, 4))

    def test_randn(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.randn([])
            out2 = paddle.randn(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2, 3, 4))

    def test_randint(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            out1 = paddle.randint(-10, 10, [])

            shape = [
                paddle.full([], 2, 'int32'),
                paddle.full([], 3, 'int32'),
                paddle.full([], 4, 'int32'),
            ]
            out2 = paddle.randint(-10, 10, shape)

            res = self.exe.run(
                paddle.static.default_main_program(), fetch_list=[out1, out2]
            )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 3, 4))

    def test_randint_like(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            out1 = paddle.rand([])
            out2 = paddle.randint_like(out1, -10, 10)

            res = self.exe.run(
                paddle.static.default_main_program(), fetch_list=[out1, out2]
            )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

    def test_standard_normal(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.standard_normal([])
            out2 = paddle.standard_normal(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2, 3, 4))

    def test_uniform(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.uniform([])
            out2 = paddle.uniform(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2, 3, 4))

    def test_empty_and_empty_like(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.empty([])
            out2 = paddle.empty_like(out1)
            out3 = paddle.empty(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2, out3]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2, 3, 4))

    def test_full_and_full_like(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.full([], 0.5)
            out2 = paddle.full_like(out1, 0.5)
            out3 = paddle.full(self.create_dynamic_shape(), 0.5)
            out4 = paddle.full(
                self.create_dynamic_shape(), paddle.full([], 0.5)
            )

            res = paddle.static.Executor().run(
                main_program,
                fetch_list=[out1, out2, out3, out4],
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2, 3, 4))
            self.assertEqual(res[3].shape, (2, 3, 4))

    def test_ones_and_ones_like(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.ones([])
            out2 = paddle.ones_like(out1)
            out3 = paddle.ones(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2, out3]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2, 3, 4))

    def test_zeros_and_zeros_like(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            out1 = paddle.zeros([])
            out2 = paddle.zeros_like(out1)
            out3 = paddle.zeros(self.create_dynamic_shape())

            res = paddle.static.Executor().run(
                main_program, fetch_list=[out1, out2, out3]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2, 3, 4))

    def test_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        w0 = paddle.arange(3, 9).reshape((3, 2)).astype(paddle.float32)
        w = paddle.to_tensor(w0, stop_gradient=False)
        emb = paddle.nn.functional.embedding(
            x=ids, weight=w, sparse=True, name="embedding"
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[emb])
        self.assertEqual(res[0].shape, (2,))
        result = [5.0, 6.0]
        for i in range(len(res)):
            self.assertEqual(res[0][i], result[i])

    def test_one_hot_label(self):
        label = paddle.full(shape=[], fill_value=2, dtype='int64')
        one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[one_hot_label])

        self.assertEqual(res[0].shape, (4,))
        self.assertEqual(res[0][2], 1)

    def test_unique_consecutive(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.rand([])
            y, inverse, counts = paddle.unique_consecutive(
                x, return_inverse=True, return_counts=True
            )

            (
                x_res,
                y_res,
                inverse_res,
                counts_res,
            ) = paddle.static.Executor().run(
                main_program, fetch_list=[x, y, inverse, counts]
            )
            self.assertEqual(x_res, y_res)
            self.assertEqual(inverse_res, 0)
            self.assertEqual(counts_res, 1)
            self.assertEqual(y_res.shape, (1,))
            self.assertEqual(inverse_res.shape, (1,))
            self.assertEqual(counts_res.shape, (1,))

    def test_unique(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.rand([])
            y, index, inverse, counts = paddle.unique(
                x, return_index=True, return_inverse=True, return_counts=True
            )

            (
                x_res,
                y_res,
                index_res,
                inverse_res,
                counts_res,
            ) = paddle.static.Executor().run(
                main_program, fetch_list=[x, y, index, inverse, counts]
            )
            self.assertEqual(x_res, y_res)
            self.assertEqual(index_res, 0)
            self.assertEqual(inverse_res, 0)
            self.assertEqual(counts_res, 1)
            self.assertEqual(y_res.shape, (1,))
            self.assertEqual(index_res.shape, (1,))
            self.assertEqual(inverse_res.shape, (1,))
            self.assertEqual(counts_res.shape, (1,))

    def test_static_matrix_rank(self):
        # 2D : OUTPUT 0D
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.eye(10)
            x.stop_gradient = False
            out = paddle.linalg.matrix_rank(x)
            exe = paddle.static.Executor()
            res = exe.run(fetch_list=[out])
            self.assertEqual(res[0].shape, ())

        # 3D : OUTPUT 1D
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            c = paddle.ones(shape=[3, 4, 5])
            c.stop_gradient = False
            out_c = paddle.linalg.matrix_rank(c)
            exe = paddle.static.Executor()
            res = exe.run(fetch_list=[out_c])
            self.assertEqual(res[0].shape, (3,))

        # 2D, tol->float : OUTPUT 0D
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_tol = paddle.eye(10)
            x_tol.stop_gradient = False
            out_tol = paddle.linalg.matrix_rank(x_tol, tol=0.1)
            exe = paddle.static.Executor()
            res = exe.run(fetch_list=[out_tol])
            self.assertEqual(res[0].shape, ())

        # 3D, tol->float : OUTPUT 1D
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            c_tol = paddle.ones(shape=[3, 4, 5])
            c_tol.stop_gradient = False
            out_c_tol = paddle.linalg.matrix_rank(c_tol, tol=0.1)
            exe = paddle.static.Executor()
            res = exe.run(fetch_list=[out_c_tol])
            self.assertEqual(res[0].shape, (3,))

        # 2D, tol->Tensor[1,2] : OUTPUT 1D
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            tol_2 = paddle.randn([2])
            d = paddle.eye(10)
            out_d = paddle.linalg.matrix_rank(d, tol=tol_2)
            exe = paddle.static.Executor()
            res = exe.run(fetch_list=[out_d])
            self.assertEqual(res[0].shape, (2,))


if __name__ == "__main__":
    unittest.main()
