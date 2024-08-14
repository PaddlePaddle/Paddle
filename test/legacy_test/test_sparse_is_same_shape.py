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

import paddle
from paddle.base.framework import in_pir_mode
from paddle.sparse.binary import is_same_shape


class TestSparseIsSameShapeAPI(unittest.TestCase):
    """
    test paddle.sparse.is_same_shape
    """

    def setUp(self):
        self.shapes = [[2, 5, 8], [3, 4]]
        self.tensors = [
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[1]),
        ]
        self.sparse_dim = 2

    def test_dense_dense(self):
        self.assertTrue(is_same_shape(self.tensors[0], self.tensors[1]))
        self.assertFalse(is_same_shape(self.tensors[0], self.tensors[2]))
        self.assertFalse(is_same_shape(self.tensors[1], self.tensors[2]))

    def test_dense_csr(self):
        self.assertTrue(
            is_same_shape(self.tensors[0], self.tensors[1].to_sparse_csr())
        )
        self.assertFalse(
            is_same_shape(self.tensors[0], self.tensors[2].to_sparse_csr())
        )
        self.assertFalse(
            is_same_shape(self.tensors[1], self.tensors[2].to_sparse_csr())
        )

    def test_dense_coo(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0], self.tensors[1].to_sparse_coo(self.sparse_dim)
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0], self.tensors[2].to_sparse_coo(self.sparse_dim)
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1], self.tensors[2].to_sparse_coo(self.sparse_dim)
            )
        )

    def test_csr_dense(self):
        self.assertTrue(
            is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[1])
        )
        self.assertFalse(
            is_same_shape(self.tensors[0].to_sparse_csr(), self.tensors[2])
        )
        self.assertFalse(
            is_same_shape(self.tensors[1].to_sparse_csr(), self.tensors[2])
        )

    def test_csr_csr(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0].to_sparse_csr(), self.tensors[1].to_sparse_csr()
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0].to_sparse_csr(), self.tensors[2].to_sparse_csr()
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1].to_sparse_csr(), self.tensors[2].to_sparse_csr()
            )
        )

    def test_csr_coo(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0].to_sparse_csr(),
                self.tensors[1].to_sparse_coo(self.sparse_dim),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0].to_sparse_csr(),
                self.tensors[2].to_sparse_coo(self.sparse_dim),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1].to_sparse_csr(),
                self.tensors[2].to_sparse_coo(self.sparse_dim),
            )
        )

    def test_coo_dense(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[1]
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim), self.tensors[2]
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1].to_sparse_coo(self.sparse_dim), self.tensors[2]
            )
        )

    def test_coo_csr(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim),
                self.tensors[1].to_sparse_csr(),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim),
                self.tensors[2].to_sparse_csr(),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1].to_sparse_coo(self.sparse_dim),
                self.tensors[2].to_sparse_csr(),
            )
        )

    def test_coo_coo(self):
        self.assertTrue(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim),
                self.tensors[1].to_sparse_coo(self.sparse_dim),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[0].to_sparse_coo(self.sparse_dim),
                self.tensors[2].to_sparse_coo(self.sparse_dim),
            )
        )
        self.assertFalse(
            is_same_shape(
                self.tensors[1].to_sparse_coo(self.sparse_dim),
                self.tensors[2].to_sparse_coo(self.sparse_dim),
            )
        )


class TestSparseIsSameShapeStatic(unittest.TestCase):
    '''
    test paddle.sparse.is_same_shape in static graph in pir mode
    only support sparse_coo_tensor in static graph
    '''

    def setUp(self):
        self.shapes = [[2, 5, 8], [3, 4]]
        self.tensors = [
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[0]),
            paddle.rand(self.shapes[1]),
        ]
        self.sparse_dim = 2

    def test_dense_dense(self):
        if in_pir_mode():
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=self.shapes[0], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=self.shapes[0], dtype='float32'
                )
                z = paddle.static.data(
                    name='z', shape=self.shapes[1], dtype='float32'
                )
                out1 = paddle.sparse.is_same_shape(x, y)
                out2 = paddle.sparse.is_same_shape(z, x)
                out3 = paddle.sparse.is_same_shape(y, z)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'x': self.tensors[0].numpy(),
                        'y': self.tensors[1].numpy(),
                        'z': self.tensors[2].numpy(),
                    },
                    fetch_list=[out1, out2, out3],
                    return_numpy=True,
                )
                self.assertTrue(fetch[0])
                self.assertFalse(fetch[1])
                self.assertFalse(fetch[2])
                paddle.disable_static()

    def test_dense_coo(self):
        if in_pir_mode():
            x_indices_data, x_values_data = (
                self.tensors[0]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[0]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            y_indices_data, y_values_data = (
                self.tensors[1]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[1]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            z_indices_data, z_values_data = (
                self.tensors[2]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[2]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=self.shapes[0], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=self.shapes[0], dtype='float32'
                )
                z = paddle.static.data(
                    name='z', shape=self.shapes[1], dtype='float32'
                )
                x_indices = paddle.static.data(
                    name='x_indices', shape=x_indices_data.shape, dtype='int64'
                )
                x_values = paddle.static.data(
                    name='x_values', shape=x_values_data.shape, dtype='float32'
                )
                y_indices = paddle.static.data(
                    name='y_indices', shape=y_indices_data.shape, dtype='int64'
                )
                y_values = paddle.static.data(
                    name='y_values', shape=y_values_data.shape, dtype='float32'
                )
                z_indices = paddle.static.data(
                    name='z_indices', shape=z_indices_data.shape, dtype='int64'
                )
                z_values = paddle.static.data(
                    name='z_values', shape=z_values_data.shape, dtype='float32'
                )
                x_coo = paddle.sparse.sparse_coo_tensor(
                    x_indices,
                    x_values,
                    shape=self.shapes[0],
                    dtype='float32',
                )
                y_coo = paddle.sparse.sparse_coo_tensor(
                    y_indices,
                    y_values,
                    shape=self.shapes[0],
                    dtype='float32',
                )
                z_coo = paddle.sparse.sparse_coo_tensor(
                    z_indices,
                    z_values,
                    shape=self.shapes[1],
                    dtype='float32',
                )
                out1 = paddle.sparse.is_same_shape(x, y_coo)
                out2 = paddle.sparse.is_same_shape(z, x_coo)
                out3 = paddle.sparse.is_same_shape(y, z_coo)
                out4 = paddle.sparse.is_same_shape(x_coo, y)
                out5 = paddle.sparse.is_same_shape(z_coo, x)
                out6 = paddle.sparse.is_same_shape(y_coo, z)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'x': self.tensors[0].numpy(),
                        'y': self.tensors[1].numpy(),
                        'z': self.tensors[2].numpy(),
                        'x_indices': x_indices_data.numpy(),
                        'x_values': x_values_data.numpy(),
                        'y_indices': y_indices_data.numpy(),
                        'y_values': y_values_data.numpy(),
                        'z_indices': z_indices_data.numpy(),
                        'z_values': z_values_data.numpy(),
                    },
                    fetch_list=[out1, out2, out3, out4, out5, out6],
                    return_numpy=True,
                )
                self.assertTrue(fetch[0])
                self.assertFalse(fetch[1])
                self.assertFalse(fetch[2])
                self.assertTrue(fetch[3])
                self.assertFalse(fetch[4])
                self.assertFalse(fetch[5])
                paddle.disable_static()

    def test_coo_coo(self):
        if in_pir_mode():
            x_indices_data, x_values_data = (
                self.tensors[0]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[0]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            y_indices_data, y_values_data = (
                self.tensors[1]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[1]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            z_indices_data, z_values_data = (
                self.tensors[2]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .indices(),
                self.tensors[2]
                .detach()
                .to_sparse_coo(self.sparse_dim)
                .values(),
            )
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_indices = paddle.static.data(
                    name='x_indices', shape=x_indices_data.shape, dtype='int64'
                )
                x_values = paddle.static.data(
                    name='x_values', shape=x_values_data.shape, dtype='float32'
                )
                y_indices = paddle.static.data(
                    name='y_indices', shape=y_indices_data.shape, dtype='int64'
                )
                y_values = paddle.static.data(
                    name='y_values', shape=y_values_data.shape, dtype='float32'
                )
                z_indices = paddle.static.data(
                    name='z_indices', shape=z_indices_data.shape, dtype='int64'
                )
                z_values = paddle.static.data(
                    name='z_values', shape=z_values_data.shape, dtype='float32'
                )
                x_coo = paddle.sparse.sparse_coo_tensor(
                    x_indices,
                    x_values,
                    shape=self.shapes[0],
                    dtype='float32',
                )
                y_coo = paddle.sparse.sparse_coo_tensor(
                    y_indices,
                    y_values,
                    shape=self.shapes[0],
                    dtype='float32',
                )
                z_coo = paddle.sparse.sparse_coo_tensor(
                    z_indices,
                    z_values,
                    shape=self.shapes[1],
                    dtype='float32',
                )
                out1 = paddle.sparse.is_same_shape(x_coo, y_coo)
                out2 = paddle.sparse.is_same_shape(z_coo, x_coo)
                out3 = paddle.sparse.is_same_shape(y_coo, z_coo)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'x_indices': x_indices_data.numpy(),
                        'x_values': x_values_data.numpy(),
                        'y_indices': y_indices_data.numpy(),
                        'y_values': y_values_data.numpy(),
                        'z_indices': z_indices_data.numpy(),
                        'z_values': z_values_data.numpy(),
                    },
                    fetch_list=[out1, out2, out3],
                    return_numpy=True,
                )
                self.assertTrue(fetch[0])
                self.assertFalse(fetch[1])
                self.assertFalse(fetch[2])
                paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
