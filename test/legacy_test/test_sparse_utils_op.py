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

import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.base.framework import in_pir_mode

devices = ['cpu', 'gpu']


class TestSparseCreate(unittest.TestCase):
    def test_create_coo_by_tensor(self):
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        dense_indices = paddle.to_tensor(indices)
        dense_elements = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(
            dense_indices, dense_elements, dense_shape, stop_gradient=False
        )
        # test the to_string.py
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_create_coo_by_np(self):
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        np.testing.assert_array_equal(3, coo.nnz())
        np.testing.assert_array_equal(indices, coo.indices().numpy())
        np.testing.assert_array_equal(values, coo.values().numpy())

    def test_create_csr_by_tensor(self):
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
            stop_gradient=stop_gradient,
        )

    def test_create_csr_by_np(self):
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
        # test the to_string.py
        np.testing.assert_array_equal(5, csr.nnz())
        np.testing.assert_array_equal(crows, csr.crows().numpy())
        np.testing.assert_array_equal(cols, csr.cols().numpy())
        np.testing.assert_array_equal(values, csr.values().numpy())

    def test_place(self):
        place = core.CPUPlace()
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        coo = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, place=place
        )
        assert coo.place.is_cpu_place()
        assert coo.values().place.is_cpu_place()
        assert coo.indices().place.is_cpu_place()

        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        csr = paddle.sparse.sparse_csr_tensor(
            crows, cols, values, [3, 5], place=place
        )
        assert csr.place.is_cpu_place()
        assert csr.crows().place.is_cpu_place()
        assert csr.cols().place.is_cpu_place()
        assert csr.values().place.is_cpu_place()

    def test_dtype(self):
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        dense_shape = [2, 2]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, dtype='float64'
        )
        assert coo.dtype == paddle.float64

        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        csr = paddle.sparse.sparse_csr_tensor(
            crows, cols, values, [3, 5], dtype='float16'
        )
        assert csr.dtype == paddle.float16

    def test_create_coo_no_shape(self):
        indices = [[0, 1], [0, 1]]
        values = [1.0, 2.0]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        coo = paddle.sparse.sparse_coo_tensor(indices, values)
        assert [2, 2] == coo.shape

    def test_create_csr_no_shape(self):
        # 2D sparse tensor
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        crows = paddle.to_tensor(crows, dtype='int32')
        cols = paddle.to_tensor(cols, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values)
        assert [3, 4] == csr.shape

        # 3D sparse tensor
        crows = [0, 2, 2, 0, 1, 1, 0, 0, 0]
        cols = [0, 1, 1]
        values = [1, 2, 5]
        crows = paddle.to_tensor(crows, dtype='int32')
        cols = paddle.to_tensor(cols, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values)
        assert [3, 2, 2] == csr.shape

        # 3D sparse tensor
        crows = [0, 1, 2, 0, 1, 1, 0, 1, 2]
        cols = [0, 2, 1, 0, 1]
        values = [1, 2, 3, 4, 5]
        crows = paddle.to_tensor(crows, dtype='int32')
        cols = paddle.to_tensor(cols, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values)
        assert [3, 2, 3] == csr.shape


class TestSparseConvert(unittest.TestCase):
    def test_to_sparse_coo(self):
        x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        dense_x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
        out = dense_x.to_sparse_coo(2)
        np.testing.assert_array_equal(out.indices().numpy(), indices)
        np.testing.assert_array_equal(out.values().numpy(), values)
        # test to_sparse_coo_grad backward
        out_grad_indices = [[0, 1], [0, 1]]
        out_grad_values = [2.0, 3.0]
        out_grad = paddle.sparse.sparse_coo_tensor(
            paddle.to_tensor(out_grad_indices),
            paddle.to_tensor(out_grad_values),
            shape=out.shape,
            stop_gradient=True,
        )
        out.backward(out_grad)
        np.testing.assert_array_equal(
            dense_x.grad.numpy(), out_grad.to_dense().numpy()
        )

    def test_coo_to_dense(self):
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices_dtypes = ['int32', 'int64']
        for indices_dtype in indices_dtypes:
            sparse_x = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices, dtype=indices_dtype),
                paddle.to_tensor(values),
                shape=[3, 4],
                stop_gradient=False,
            )
            sparse_x.retain_grads()
            dense_tensor = sparse_x.to_dense()
            # test to_dense_grad backward
            out_grad = [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
            dense_tensor.backward(paddle.to_tensor(out_grad))
            # mask the out_grad by sparse_x.indices()
            correct_x_grad = [2.0, 4.0, 7.0, 9.0, 10.0]
            np.testing.assert_array_equal(
                correct_x_grad, sparse_x.grad.values().numpy()
            )

            paddle.device.set_device("cpu")
            sparse_x_cpu = paddle.sparse.sparse_coo_tensor(
                paddle.to_tensor(indices, dtype=indices_dtype),
                paddle.to_tensor(values),
                shape=[3, 4],
                stop_gradient=False,
            )
            sparse_x_cpu.retain_grads()
            dense_tensor_cpu = sparse_x_cpu.to_dense()
            dense_tensor_cpu.backward(paddle.to_tensor(out_grad))
            np.testing.assert_array_equal(
                correct_x_grad, sparse_x_cpu.grad.values().numpy()
            )

    def test_to_sparse_csr(self):
        x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_x = paddle.to_tensor(x)
        out = dense_x.to_sparse_csr()
        np.testing.assert_array_equal(out.crows().numpy(), crows)
        np.testing.assert_array_equal(out.cols().numpy(), cols)
        np.testing.assert_array_equal(out.values().numpy(), values)

        dense_tensor = out.to_dense()
        np.testing.assert_array_equal(dense_tensor.numpy(), x)

    def test_coo_values_grad(self):
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sparse_x = paddle.sparse.sparse_coo_tensor(
            paddle.to_tensor(indices),
            paddle.to_tensor(values),
            shape=[3, 4],
            stop_gradient=False,
        )
        sparse_x.retain_grads()
        values_tensor = sparse_x.values()
        out_grad = [2.0, 3.0, 5.0, 8.0, 9.0]
        # test coo_values_grad
        values_tensor.backward(paddle.to_tensor(out_grad))
        np.testing.assert_array_equal(out_grad, sparse_x.grad.values().numpy())
        indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
        values = [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
        sparse_x = paddle.sparse.sparse_coo_tensor(
            paddle.to_tensor(indices),
            paddle.to_tensor(values),
            shape=[3, 4, 2],
            stop_gradient=False,
        )
        sparse_x.retain_grads()
        values_tensor = sparse_x.values()
        out_grad = [
            [2.0, 2.0],
            [3.0, 3.0],
            [5.0, 5.0],
            [8.0, 8.0],
            [9.0, 9.0],
        ]
        # test coo_values_grad
        values_tensor.backward(paddle.to_tensor(out_grad))
        np.testing.assert_array_equal(out_grad, sparse_x.grad.values().numpy())

    def test_sparse_coo_tensor_grad(self):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                paddle.device.set_device(device)
                indices = [[0, 1], [0, 1]]
                values = [1, 2]
                indices = paddle.to_tensor(indices, dtype='int32')
                values = paddle.to_tensor(
                    values, dtype='float32', stop_gradient=False
                )
                sparse_x = paddle.sparse.sparse_coo_tensor(
                    indices, values, shape=[2, 2], stop_gradient=False
                )
                grad_indices = [[0, 1], [1, 1]]
                grad_values = [2, 3]
                grad_indices = paddle.to_tensor(grad_indices, dtype='int32')
                grad_values = paddle.to_tensor(grad_values, dtype='float32')
                sparse_out_grad = paddle.sparse.sparse_coo_tensor(
                    grad_indices, grad_values, shape=[2, 2]
                )
                sparse_x.backward(sparse_out_grad)
                correct_values_grad = [0, 3]
                np.testing.assert_array_equal(
                    correct_values_grad, values.grad.numpy()
                )

                # test the non-zero values is a vector
                values = [[1, 1], [2, 2]]
                values = paddle.to_tensor(
                    values, dtype='float32', stop_gradient=False
                )
                sparse_x = paddle.sparse.sparse_coo_tensor(
                    indices, values, shape=[2, 2, 2], stop_gradient=False
                )
                grad_values = [[2, 2], [3, 3]]
                grad_values = paddle.to_tensor(grad_values, dtype='float32')
                sparse_out_grad = paddle.sparse.sparse_coo_tensor(
                    grad_indices, grad_values, shape=[2, 2, 2]
                )
                sparse_x.backward(sparse_out_grad)
                correct_values_grad = [[0, 0], [3, 3]]
                np.testing.assert_array_equal(
                    correct_values_grad, values.grad.numpy()
                )

    def test_sparse_coo_tensor_sorted(self):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                paddle.device.set_device(device)
                # test unsorted and duplicate indices
                indices = [[1, 0, 0], [0, 1, 1]]
                values = [1.0, 2.0, 3.0]
                indices = paddle.to_tensor(indices, dtype='int32')
                values = paddle.to_tensor(values, dtype='float32')
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                sparse_x = paddle.sparse.coalesce(sparse_x)
                indices_sorted = [[0, 1], [1, 0]]
                values_sorted = [5.0, 1.0]
                np.testing.assert_array_equal(
                    indices_sorted, sparse_x.indices().numpy()
                )
                np.testing.assert_array_equal(
                    values_sorted, sparse_x.values().numpy()
                )

                # test the non-zero values is a vector
                values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
                values = paddle.to_tensor(values, dtype='float32')
                sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)
                sparse_x = paddle.sparse.coalesce(sparse_x)
                values_sorted = [[5.0, 5.0], [1.0, 1.0]]
                np.testing.assert_array_equal(
                    indices_sorted, sparse_x.indices().numpy()
                )
                np.testing.assert_array_equal(
                    values_sorted, sparse_x.values().numpy()
                )

    def test_batch_csr(self):
        def verify(dense_x):
            sparse_x = dense_x.to_sparse_csr()
            out = sparse_x.to_dense()
            np.testing.assert_allclose(out.numpy(), dense_x.numpy())

        shape = np.random.randint(low=1, high=10, size=3)
        shape = list(shape)
        dense_x = paddle.randn(shape)
        dense_x = paddle.nn.functional.dropout(dense_x, p=0.5)
        verify(dense_x)

        # test batches=1
        shape[0] = 1
        dense_x = paddle.randn(shape)
        dense_x = paddle.nn.functional.dropout(dense_x, p=0.5)
        verify(dense_x)

        shape = np.random.randint(low=3, high=10, size=3)
        shape = list(shape)
        dense_x = paddle.randn(shape)
        # set the 0th batch to zero
        dense_x[0] = 0
        verify(dense_x)

        dense_x = paddle.randn(shape)
        # set the 1st batch to zero
        dense_x[1] = 0
        verify(dense_x)

        dense_x = paddle.randn(shape)
        # set the 2nd batch to zero
        dense_x[2] = 0
        verify(dense_x)

    def test_zero_nnz(self):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                paddle.device.set_device(device)
                x1 = paddle.zeros([2, 2, 2])
                x2 = paddle.zeros([2, 2, 2])
                sp_csr_x = x1.to_sparse_csr()
                sp_coo_x = x2.to_sparse_coo(1)
                sp_coo_x.stop_gradient = False

                out1 = sp_csr_x.to_dense()
                out2 = sp_coo_x.to_dense()
                out2.backward()
                np.testing.assert_allclose(out1.numpy(), x1.numpy())
                np.testing.assert_allclose(out2.numpy(), x2.numpy())
                np.testing.assert_allclose(
                    sp_coo_x.grad.to_dense().numpy().sum(), 0.0
                )


class TestCooError(unittest.TestCase):
    def test_small_shape(self):
        with self.assertRaises(ValueError):
            indices = [[2, 3], [0, 2]]
            values = [1, 2]
            # 1. the shape too small
            dense_shape = [2, 2]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                indices, values, shape=dense_shape
            )

    def test_same_nnz(self):
        with self.assertRaises(ValueError):
            # 2. test the nnz of indices must same as nnz of values
            indices = [[1, 2], [1, 0]]
            values = [1, 2, 3]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)

    def test_same_dimensions(self):
        with self.assertRaises(ValueError):
            indices = [[1, 2], [1, 0]]
            values = [1, 2, 3]
            shape = [2, 3, 4]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                indices, values, shape=shape
            )

    def test_indices_dtype(self):
        with self.assertRaises(TypeError):
            indices = [[1.0, 2.0], [0, 1]]
            values = [1, 2]
            sparse_x = paddle.sparse.sparse_coo_tensor(indices, values)


class TestCsrError(unittest.TestCase):
    def test_dimension1(self):
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_dimension2(self):
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3, 3, 3, 3]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_same_shape1(self):
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2, 3]
            values = [1, 2, 3]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_same_shape2(self):
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3]
            cols = [0, 1, 2, 3]
            values = [1, 2, 3, 4]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_same_shape3(self):
        with self.assertRaises(ValueError):
            crows = [0, 1, 2, 3, 0, 1, 2]
            cols = [0, 1, 2, 3, 0, 1, 2]
            values = [1, 2, 3, 4, 0, 1, 2]
            shape = [2, 3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_crows_first_value(self):
        with self.assertRaises(ValueError):
            crows = [1, 1, 2, 3]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3, 4]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_dtype(self):
        with self.assertRaises(TypeError):
            crows = [0, 1, 2, 3.0]
            cols = [0, 1, 2]
            values = [1, 2, 3]
            shape = [3]
            sparse_x = paddle.sparse.sparse_csr_tensor(
                crows, cols, values, shape
            )

    def test_error_crows(self):
        with self.assertRaises(ValueError):
            crows = [0, 2, 2, 0, 1, 1, 0, 0, 0, 0]
            cols = [0, 1, 1]
            values = [1, 2, 5]
            crows = paddle.to_tensor(crows, dtype='int32')
            cols = paddle.to_tensor(cols, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            coo = paddle.sparse.sparse_csr_tensor(crows, cols, values)


devices = []
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())
else:
    devices.append('cpu')


class TestSparseCoalesceStatic(unittest.TestCase):
    '''
    test the coalesce function in static graph in pir mode
    '''

    def sort_and_merge(self, indices, values):
        '''
        sort the indices and merge the duplicate values in the same indices, using numpy and provide the correct result
        '''
        indices = np.array(indices)
        values = np.array(values)
        indices = indices[:, np.lexsort((indices[1], indices[0]))]
        unique_indices, unique_indices_idx = np.unique(
            indices, axis=1, return_index=True
        )
        v = []
        for interval in zip(
            unique_indices_idx.tolist(),
            unique_indices_idx.tolist()[1:] + [None],
        ):
            v.append(np.sum(values[interval[0] : interval[1]]))
        unique_values = np.array(v)
        return unique_indices, unique_values

    def check_result(self, indices, values):
        for device in devices:
            paddle.device.set_device(device)
            indices_tensor = paddle.to_tensor(indices, dtype='int32')
            values_tensor = paddle.to_tensor(values, dtype='float32')
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_indices = paddle.static.data(
                    name="x_indices",
                    shape=indices_tensor.shape,
                    dtype=indices_tensor.dtype,
                )
                x_values = paddle.static.data(
                    name="x_values",
                    shape=values_tensor.shape,
                    dtype=values_tensor.dtype,
                )
                sp_x = paddle.sparse.sparse_coo_tensor(
                    x_indices,
                    x_values,
                    dtype=x_values.dtype,
                )
                sp_x = paddle.sparse.coalesce(sp_x)

                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        "x_indices": indices_tensor.numpy(),
                        "x_values": values_tensor.numpy(),
                    },
                    fetch_list=[sp_x.indices(), sp_x.values()],
                    return_numpy=True,
                )
                unique_indices, unique_values = self.sort_and_merge(
                    indices, values
                )
                np.testing.assert_array_equal(fetch[0], unique_indices)
                np.testing.assert_array_equal(fetch[1], unique_values)
                paddle.disable_static()

    def test_sparse_coalesce(self):
        indices = [[0, 1, 1], [0, 1, 1]]
        values = [1.0, 2.0, 3.0]
        if in_pir_mode():
            self.check_result(indices, values)

        indices = [[0, 1, 1], [0, 1, 1]]
        values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        if in_pir_mode():
            self.check_result(indices, values)


if __name__ == "__main__":
    unittest.main()
