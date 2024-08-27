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


class TestReshape(unittest.TestCase):
    """
    Test the API paddle.sparse.reshape on some sparse tensors.
    x: sparse, out: sparse
    """

    def check_result(self, x_shape, new_shape, format):
        """
        x_shape: original shape
        new_shape: new shape
        format: "coo" or "csr"
        Transform a sparse tensor with shape "x_shape" to
        a sparse tensor with shape "new_shape".
        Compare the output of paddle.reshape and the output of
        paddle.sparse.reshape.
        """
        mask = np.random.randint(0, 2, x_shape)
        while np.sum(mask) == 0:
            mask = paddle.randint(0, 2, x_shape)
        np_x = np.random.randint(-100, 100, x_shape) * mask

        # check cpu kernel
        dense_x = paddle.to_tensor(np_x, place=paddle.CPUPlace())
        dense_x.stop_gradient = False
        dense_out = paddle.reshape(dense_x, new_shape)

        if format == "coo":
            sp_x = paddle.to_tensor(
                np_x, place=paddle.CPUPlace()
            ).to_sparse_coo(len(x_shape))
        else:
            sp_x = paddle.to_tensor(
                np_x, place=paddle.CPUPlace()
            ).to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.reshape(sp_x, new_shape)

        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            dense_x.grad.numpy() * np_x.astype('bool').astype('int'),
            rtol=1e-05,
        )

        # check gpu kernel
        if paddle.device.is_compiled_with_cuda():
            dense_x = paddle.to_tensor(np_x, place=paddle.CUDAPlace(0))
            dense_x.stop_gradient = False
            dense_out = paddle.reshape(dense_x, new_shape)

            if format == "coo":
                sp_x = paddle.to_tensor(
                    np_x, place=paddle.CUDAPlace(0)
                ).to_sparse_coo(len(x_shape))
            else:
                sp_x = paddle.to_tensor(
                    np_x, place=paddle.CUDAPlace(0)
                ).to_sparse_csr()
            sp_x.stop_gradient = False
            sp_out = paddle.sparse.reshape(sp_x, new_shape)

            np.testing.assert_allclose(
                sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
            )

            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(
                sp_x.grad.to_dense().numpy(),
                dense_x.grad.numpy() * np_x.astype('bool').astype('int'),
                rtol=1e-05,
            )

    def test_reshape_2d(self):
        self.check_result(
            [2, 5],
            [
                10,
            ],
            'coo',
        )
        self.check_result([12, 5], [15, 4], 'coo')

        self.check_result([10, 5], [2, 25], 'csr')
        self.check_result([9, 8], [18, 4], 'csr')

    def test_reshape_3d(self):
        self.check_result([6, 2, 3], [6, 2, 3], 'coo')
        self.check_result([6, 2, 3], [2, 3, 3, 2], 'coo')
        self.check_result([6, 2, 3], [1, 18, 2], 'coo')
        self.check_result([6, 2, 3], [2, 9, 2], 'coo')
        self.check_result([6, 2, 3], [2, 1, 18], 'coo')
        self.check_result([6, 2, 3], [1, 2, 2, 3, 3], 'coo')

        self.check_result([6, 2, 3], [6, 2, 3], 'csr')
        self.check_result([6, 2, 3], [6, 3, 2], 'csr')
        self.check_result([6, 2, 3], [2, 6, 3], 'csr')
        self.check_result([6, 2, 3], [3, 6, 2], 'csr')
        self.check_result([6, 2, 3], [4, 9, 1], 'csr')
        self.check_result([6, 2, 3], [12, 1, 3], 'csr')

    def test_reshape_nd(self):
        self.check_result([8, 3, 4, 4, 5, 3], [24, 8, 10, 3], 'coo')
        self.check_result([3, 4, 4, 5, 7], [1, 12, 2, 5, 14], 'coo')

    def test_reshape_with_zero_or_minus_one_in_new_shape(self):
        self.check_result([6, 2, 3], [-1, 0, 3], 'coo')
        self.check_result([6, 2, 3], [2, 3, 0, -1], 'coo')
        self.check_result([6, 2, 3], [1, -1, 2], 'coo')
        self.check_result([6, 2, 3], [-1, 9, 2], 'coo')
        self.check_result([6, 2, 3], [2, -1, 18], 'coo')
        self.check_result([6, 2, 3], [1, 0, 2, -1, 3], 'coo')

        self.check_result([6, 2, 3], [0, 0, -1], 'csr')
        self.check_result([6, 2, 3], [-1, 3, 2], 'csr')
        self.check_result([6, 2, 3], [2, -1, 0], 'csr')
        self.check_result([6, 2, 3], [-1, 6, 2], 'csr')
        self.check_result([6, 2, 3], [-1, 9, 1], 'csr')
        self.check_result([6, 2, 3], [-1, 1, 3], 'csr')


devices = []
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())
else:
    devices.append('cpu')


class TestSparseReshapeStatic(unittest.TestCase):
    """
    Test the API paddle.sparse.reshape on some sparse tensors. static graph
    x: sparse, out: sparse
    """

    def check_result_coo(self, x_shape, new_shape):
        """
        x_shape: original shape
        new_shape: new shape
        static graph only supports coo format.
        Transform a sparse tensor with shape "x_shape" to
        a sparse tensor with shape "new_shape".
        Compare the output of paddle.reshape and the output of
        paddle.sparse.reshape.
        """
        for device in devices:
            paddle.device.set_device(device)
            mask = paddle.randint(0, 2, x_shape)
            n = 0
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, x_shape)
                n += 1
                if n > 10000:
                    mask[0] = 1
                    break
            origin_data = (
                paddle.rand(x_shape, dtype='float32') + 1
            ) * mask.astype('float32')
            indices_data, values_data = (
                origin_data.detach()
                .to_sparse_coo(sparse_dim=len(x_shape))
                .indices(),
                origin_data.detach()
                .to_sparse_coo(sparse_dim=len(x_shape))
                .values(),
            )

            dense_x = origin_data
            dense_x.stop_gradient = False
            dense_out = paddle.reshape(dense_x, new_shape)

            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                indices = paddle.static.data(
                    name='indices',
                    shape=indices_data.shape,
                    dtype=indices_data.dtype,
                )
                values = paddle.static.data(
                    name='values',
                    shape=values_data.shape,
                    dtype=values_data.dtype,
                )
                sp_x = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=dense_x.shape,
                    dtype=dense_x.dtype,
                )

                sp_out = paddle.sparse.reshape(sp_x, new_shape)
                sp_dense_out = sp_out.to_dense()

                sparse_exe = paddle.static.Executor()
                sparse_fetch = sparse_exe.run(
                    feed={
                        'indices': indices_data.numpy(),
                        "values": values_data.numpy(),
                    },
                    fetch_list=[sp_dense_out],
                    return_numpy=True,
                )

                np.testing.assert_allclose(
                    dense_out.numpy(), sparse_fetch[0], rtol=1e-5
                )
                paddle.disable_static()

    def test_reshape_2d(self):
        self.check_result_coo(
            [2, 5],
            [
                10,
            ],
        )
        self.check_result_coo([12, 5], [15, 4])

        self.check_result_coo([10, 5], [2, 25])
        self.check_result_coo([9, 8], [18, 4])

    def test_reshape_3d(self):
        self.check_result_coo([6, 2, 3], [6, 2, 3])
        self.check_result_coo([6, 2, 3], [2, 3, 3, 2])
        self.check_result_coo([6, 2, 3], [1, 18, 2])
        self.check_result_coo([6, 2, 3], [2, 9, 2])
        self.check_result_coo([6, 2, 3], [2, 1, 18])
        self.check_result_coo([6, 2, 3], [1, 2, 2, 3, 3])

        self.check_result_coo([6, 2, 3], [6, 2, 3])
        self.check_result_coo([6, 2, 3], [6, 3, 2])
        self.check_result_coo([6, 2, 3], [2, 6, 3])
        self.check_result_coo([6, 2, 3], [3, 6, 2])
        self.check_result_coo([6, 2, 3], [4, 9, 1])
        self.check_result_coo([6, 2, 3], [12, 1, 3])

    def test_reshape_nd(self):
        self.check_result_coo([8, 3, 4, 4, 5, 3], [24, 8, 10, 3])
        self.check_result_coo([3, 4, 4, 5, 7], [1, 12, 2, 5, 14])

    def test_reshape_with_zero_or_minus_one_in_new_shape(self):
        self.check_result_coo([6, 2, 3], [-1, 0, 3])
        self.check_result_coo([6, 2, 3], [2, 3, 0, -1])
        self.check_result_coo([6, 2, 3], [1, -1, 2])
        self.check_result_coo([6, 2, 3], [-1, 9, 2])
        self.check_result_coo([6, 2, 3], [2, -1, 18])
        self.check_result_coo([6, 2, 3], [1, 0, 2, -1, 3])

        self.check_result_coo([6, 2, 3], [0, 0, -1])
        self.check_result_coo([6, 2, 3], [-1, 3, 2])
        self.check_result_coo([6, 2, 3], [2, -1, 0])
        self.check_result_coo([6, 2, 3], [-1, 6, 2])
        self.check_result_coo([6, 2, 3], [-1, 9, 1])
        self.check_result_coo([6, 2, 3], [-1, 1, 3])


if __name__ == "__main__":
    unittest.main()
