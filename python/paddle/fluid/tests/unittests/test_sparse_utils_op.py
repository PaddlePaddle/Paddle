# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard

devices = ['cpu', 'gpu']


class TestSparseCreate(unittest.TestCase):
    def test_create_coo_by_tensor(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_indices = paddle.to_tensor(indices)
            dense_elements = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(
                dense_indices, dense_elements, dense_shape, stop_gradient=False)
            # test the to_string.py
            print(coo)
            assert np.array_equal(indices, coo.indices().numpy())
            assert np.array_equal(values, coo.values().numpy())

    def test_create_coo_by_np(self):
        with _test_eager_guard():
            indices = [[0, 1, 2], [1, 2, 0]]
            values = [1.0, 2.0, 3.0]
            dense_shape = [3, 3]
            coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            assert np.array_equal(indices, coo.indices().numpy())
            assert np.array_equal(values, coo.values().numpy())

    def test_create_csr_by_tensor(self):
        with _test_eager_guard():
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_crows = paddle.to_tensor(crows)
            dense_cols = paddle.to_tensor(cols)
            dense_elements = paddle.to_tensor(values, dtype='float32')
            stop_gradient = False
            csr = paddle.sparse.sparse_csr_tensor(
                dense_crows,
                dense_cols,
                dense_elements,
                dense_shape,
                stop_gradient=stop_gradient)

    def test_create_csr_by_np(self):
        with _test_eager_guard():
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            csr = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                  dense_shape)
            # test the to_string.py
            print(csr)
            assert np.array_equal(crows, csr.crows().numpy())
            assert np.array_equal(cols, csr.cols().numpy())
            assert np.array_equal(values, csr.values().numpy())

    def test_place(self):
        with _test_eager_guard():
            place = core.CPUPlace()
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            dense_shape = [2, 2]
            coo = paddle.sparse.sparse_coo_tensor(
                indices, values, dense_shape, place=place)
            assert coo.place.is_cpu_place()
            assert coo.values().place.is_cpu_place()
            assert coo.indices().place.is_cpu_place()

            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            csr = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, [3, 5], place=place)
            assert csr.place.is_cpu_place()
            assert csr.crows().place.is_cpu_place()
            assert csr.cols().place.is_cpu_place()
            assert csr.values().place.is_cpu_place()

    def test_dtype(self):
        with _test_eager_guard():
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            dense_shape = [2, 2]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(
                indices, values, dense_shape, dtype='float64')
            assert coo.dtype == paddle.float64

            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            csr = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, [3, 5], dtype='float16')
            assert csr.dtype == paddle.float16

    def test_create_coo_no_shape(self):
        with _test_eager_guard():
            indices = [[0, 1], [0, 1]]
            values = [1.0, 2.0]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_coo_tensor(indices, values)
            assert [2, 2] == coo.shape


class TestSparseConvert(unittest.TestCase):
    def test_to_sparse_coo(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            dense_x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
            out = dense_x.to_sparse_coo(2)
            assert np.array_equal(out.indices().numpy(), indices)
            assert np.array_equal(out.values().numpy(), values)
            #test to_sparse_coo_grad backward
            out_grad_indices = [[0, 1], [0, 1]]
            out_grad_values = [2.0, 3.0]
            out_grad = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(out_grad_indices),
                paddle.to_tensor(out_grad_values),
                shape=out.shape,
                stop_gradient=True)
            out.backward(out_grad)
            assert np.array_equal(dense_x.grad.numpy(),
                                  out_grad.to_dense().numpy())

    def test_coo_to_dense(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values),
                shape=[3, 4],
                stop_gradient=False)
            dense_tensor = sparse_x.to_dense()
            #test to_dense_grad backward
            out_grad = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]
            dense_tensor.backward(paddle.to_tensor(out_grad))
            #mask the out_grad by sparse_x.indices() 
            correct_x_grad = [2.0, 4.0, 7.0, 9.0, 10.0]
            assert np.array_equal(correct_x_grad,
                                  sparse_x.grad.values().numpy())

            paddle.device.set_device("cpu")
            sparse_x_cpu = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values),
                shape=[3, 4],
                stop_gradient=False)
            dense_tensor_cpu = sparse_x_cpu.to_dense()
            dense_tensor_cpu.backward(paddle.to_tensor(out_grad))
            assert np.array_equal(correct_x_grad,
                                  sparse_x_cpu.grad.values().numpy())

    def test_to_sparse_csr(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = dense_x.to_sparse_csr()
            assert np.array_equal(out.crows().numpy(), crows)
            assert np.array_equal(out.cols().numpy(), cols)
            assert np.array_equal(out.values().numpy(), values)

            dense_tensor = out.to_dense()
            assert np.array_equal(dense_tensor.numpy(), x)

    def test_coo_values_grad(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values),
                shape=[3, 4],
                stop_gradient=False)
            values_tensor = sparse_x.values()
            out_grad = [2.0, 3.0, 5.0, 8.0, 9.0]
            # test coo_values_grad
            values_tensor.backward(paddle.to_tensor(out_grad))
            assert np.array_equal(out_grad, sparse_x.grad.values().numpy())
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
                      [5.0, 5.0]]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values),
                shape=[3, 4, 2],
                stop_gradient=False)
            values_tensor = sparse_x.values()
            out_grad = [[2.0, 2.0], [3.0, 3.0], [5.0, 5.0], [8.0, 8.0],
                        [9.0, 9.0]]
            # test coo_values_grad
            values_tensor.backward(paddle.to_tensor(out_grad))
            assert np.array_equal(out_grad, sparse_x.grad.values().numpy())

    def test_sparse_coo_tensor_grad(self):
        with _test_eager_guard():
            for device in devices:
                if device == 'cpu' or (device == 'gpu' and
                                       paddle.is_compiled_with_cuda()):
                    paddle.device.set_device(device)
                    indices = [[0, 1], [0, 1]]
                    values = [1, 2]
                    indices = paddle.to_tensor(indices, dtype='int32')
                    values = paddle.to_tensor(
                        values, dtype='float32', stop_gradient=False)
                    sparse_x = paddle.sparse.sparse_coo_tensor(
                        indices, values, shape=[2, 2], stop_gradient=False)
                    grad_indices = [[0, 1], [1, 1]]
                    grad_values = [2, 3]
                    grad_indices = paddle.to_tensor(grad_indices, dtype='int32')
                    grad_values = paddle.to_tensor(grad_values, dtype='float32')
                    sparse_out_grad = paddle.sparse.sparse_coo_tensor(
                        grad_indices, grad_values, shape=[2, 2])
                    sparse_x.backward(sparse_out_grad)
                    correct_values_grad = [0, 3]
                    assert np.array_equal(correct_values_grad,
                                          values.grad.numpy())

                    # test the non-zero values is a vector
                    values = [[1, 1], [2, 2]]
                    values = paddle.to_tensor(
                        values, dtype='float32', stop_gradient=False)
                    sparse_x = paddle.sparse.sparse_coo_tensor(
                        indices, values, shape=[2, 2, 2], stop_gradient=False)
                    grad_values = [[2, 2], [3, 3]]
                    grad_values = paddle.to_tensor(grad_values, dtype='float32')
                    sparse_out_grad = paddle.sparse.sparse_coo_tensor(
                        grad_indices, grad_values, shape=[2, 2, 2])
                    sparse_x.backward(sparse_out_grad)
                    correct_values_grad = [[0, 0], [3, 3]]
                    assert np.array_equal(correct_values_grad,
                                          values.grad.numpy())

    def test_sparse_coo_tensor_sorted(self):
        with _test_eager_guard():
            for device in devices:
                if device == 'cpu' or (device == 'gpu' and
                                       paddle.is_compiled_with_cuda()):
                    paddle.device.set_device(device)
                    #test unsorted and duplicate indices 
                    indices = [[1, 0, 0], [0, 1, 1]]
                    values = [1.0, 2.0, 3.0]
                    indices = paddle.to_tensor(indices, dtype='int32')
                    values = paddle.to_tensor(values, dtype='float32')
                    sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                    indices_sorted = [[0, 1], [1, 0]]
                    values_sorted = [5.0, 1.0]
                    assert np.array_equal(indices_sorted,
                                          sparse_x.indices().numpy())
                    assert np.array_equal(values_sorted,
                                          sparse_x.values().numpy())

                    # test the non-zero values is a vector
                    values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
                    values = paddle.to_tensor(values, dtype='float32')
                    sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                    values_sorted = [[5.0, 5.0], [1.0, 1.0]]
                    assert np.array_equal(indices_sorted,
                                          sparse_x.indices().numpy())
                    assert np.array_equal(values_sorted,
                                          sparse_x.values().numpy())


class TestCooError(unittest.TestCase):
    def test_small_shape(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                indices = [[2, 3], [0, 2]]
                values = [1, 2]
                # 1. the shape too small
                dense_shape = [2, 2]
                sparse_x = paddle.sparse.sparse_coo_tensor(
                    indices, values, shape=dense_shape)

    def test_same_nnz(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                # 2. test the nnz of indices must same as nnz of values
                indices = [[1, 2], [1, 0]]
                values = [1, 2, 3]
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)

    def test_same_dimensions(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                indices = [[1, 2], [1, 0]]
                values = [1, 2, 3]
                shape = [2, 3, 4]
                sparse_x = paddle.sparse.sparse_coo_tensor(
                    indices, values, shape=shape)

    def test_indices_dtype(self):
        with _test_eager_guard():
            with self.assertRaises(TypeError):
                indices = [[1.0, 2.0], [0, 1]]
                values = [1, 2]
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)


class TestCsrError(unittest.TestCase):
    def test_dimension1(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [0, 1, 2, 3]
                cols = [0, 1, 2]
                values = [1, 2, 3]
                shape = [3]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_dimension2(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [0, 1, 2, 3]
                cols = [0, 1, 2]
                values = [1, 2, 3]
                shape = [3, 3, 3, 3]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_same_shape1(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [0, 1, 2, 3]
                cols = [0, 1, 2, 3]
                values = [1, 2, 3]
                shape = [3, 4]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_same_shape2(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [0, 1, 2, 3]
                cols = [0, 1, 2, 3]
                values = [1, 2, 3, 4]
                shape = [3, 4]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_same_shape3(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [0, 1, 2, 3, 0, 1, 2]
                cols = [0, 1, 2, 3, 0, 1, 2]
                values = [1, 2, 3, 4, 0, 1, 2]
                shape = [2, 3, 4]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_crows_first_value(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                crows = [1, 1, 2, 3]
                cols = [0, 1, 2]
                values = [1, 2, 3]
                shape = [3, 4]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)

    def test_dtype(self):
        with _test_eager_guard():
            with self.assertRaises(TypeError):
                crows = [0, 1, 2, 3.0]
                cols = [0, 1, 2]
                values = [1, 2, 3]
                shape = [3]
                sparse_x = paddle.sparse.sparse_csr_tensor(crows, cols, values,
                                                           shape)


if __name__ == "__main__":
    unittest.main()
