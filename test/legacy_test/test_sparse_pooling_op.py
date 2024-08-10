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

import copy
import unittest

import numpy as np

import paddle
from paddle.base.framework import in_pir_mode


class TestMaxPool3DFunc(unittest.TestCase):
    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 4, 4, 4, 4))

    def setKernelSize(self):
        self.kernel_sizes = [3, 3, 3]

    def setStride(self):
        self.strides = [1, 1, 1]

    def setPadding(self):
        self.paddings = [0, 0, 0]

    def setUp(self):
        self.setInput()
        self.setKernelSize()
        self.setStride()
        self.setPadding()

    def test(self):
        self.setUp()
        self.dense_x.stop_gradient = False
        sparse_x = self.dense_x.to_sparse_coo(4)
        sparse_out = paddle.sparse.nn.functional.max_pool3d(
            sparse_x,
            self.kernel_sizes,
            stride=self.strides,
            padding=self.paddings,
        )
        out = sparse_out.to_dense()
        out.backward(out)

        dense_x = copy.deepcopy(self.dense_x)
        dense_out = paddle.nn.functional.max_pool3d(
            dense_x,
            self.kernel_sizes,
            stride=self.strides,
            padding=self.paddings,
            data_format='NDHWC',
        )
        dense_out.backward(dense_out)

        # compare with dense
        np.testing.assert_allclose(dense_out.numpy(), out.numpy())
        np.testing.assert_allclose(
            dense_x.grad.numpy(), self.dense_x.grad.numpy()
        )


class TestStride(TestMaxPool3DFunc):
    def setStride(self):
        self.strides = 1


class TestPadding(TestMaxPool3DFunc):
    def setPadding(self):
        self.paddings = 1

    def setInput(self):
        self.dense_x = paddle.randn((1, 5, 6, 8, 3))


class TestKernelSize(TestMaxPool3DFunc):
    def setKernelSize(self):
        self.kernel_sizes = [5, 5, 5]

    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 6, 9, 6, 3))


class TestInput(TestMaxPool3DFunc):
    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((2, 6, 7, 9, 3))
        dropout = paddle.nn.Dropout(0.8)
        self.dense_x = dropout(self.dense_x)


class TestMaxPool3DAPI(unittest.TestCase):
    def test(self):
        dense_x = paddle.randn((2, 3, 6, 6, 3))
        sparse_x = dense_x.to_sparse_coo(4)
        max_pool3d = paddle.sparse.nn.MaxPool3D(
            kernel_size=3, data_format='NDHWC'
        )
        out = max_pool3d(sparse_x)
        out = out.to_dense()

        dense_out = paddle.nn.functional.max_pool3d(
            dense_x, 3, data_format='NDHWC'
        )
        np.testing.assert_allclose(dense_out.numpy(), out.numpy())


devices = []
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())
else:
    devices.append('cpu')


class TestMaxPool3DAPIStatic(unittest.TestCase):
    '''
    Test MaxPool3D API with static graph mode in pir mode.
    '''

    def setInput(self):
        self.dense_x = paddle.randn((1, 4, 4, 4, 3))

    def setKernelSize(self):
        self.kernel_sizes = [3, 3, 3]

    def setStride(self):
        self.strides = [1, 1, 1]

    def setPadding(self):
        self.paddings = [0, 0, 0]

    def setUp(self):
        self.setInput()
        self.setKernelSize()
        self.setStride()
        self.setPadding()

    def test(self):
        if in_pir_mode():
            self.setUp()
            for device in devices:
                paddle.set_device(device)
                x_indices_data, x_values_data = (
                    self.dense_x.detach().to_sparse_coo(sparse_dim=4).indices(),
                    self.dense_x.detach().to_sparse_coo(sparse_dim=4).values(),
                )
                dense_out = paddle.nn.functional.max_pool3d(
                    self.dense_x,
                    self.kernel_sizes,
                    stride=self.strides,
                    padding=self.paddings,
                    data_format='NDHWC',
                )

                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_indices = paddle.static.data(
                        name="x_indices",
                        shape=x_indices_data.shape,
                        dtype=x_indices_data.dtype,
                    )
                    x_values = paddle.static.data(
                        name="x_values",
                        shape=x_values_data.shape,
                        dtype=x_values_data.dtype,
                    )
                    static_x = paddle.sparse.sparse_coo_tensor(
                        x_indices,
                        x_values,
                        shape=self.dense_x.shape,
                        dtype=self.dense_x.dtype,
                    )
                    sparse_out = paddle.sparse.nn.functional.max_pool3d(
                        static_x,
                        self.kernel_sizes,
                        stride=self.strides,
                        padding=self.paddings,
                    )
                    out = sparse_out.to_dense()
                    exe = paddle.static.Executor()
                    sp_fetch = exe.run(
                        feed={
                            "x_indices": x_indices_data.numpy(),
                            "x_values": x_values_data.numpy(),
                        },
                        fetch_list=[out],
                        return_numpy=True,
                    )
                    np.testing.assert_allclose(
                        dense_out.numpy(), sp_fetch[0], rtol=1e-05
                    )
                    paddle.disable_static()


class TestStrideStatic(TestMaxPool3DAPIStatic):
    def setStride(self):
        self.strides = 1


class TestPaddingStatic(TestMaxPool3DAPIStatic):
    def setPadding(self):
        self.paddings = 1

    def setInput(self):
        self.dense_x = paddle.randn((1, 5, 6, 8, 3))


class TestKernelSizeStatic(TestMaxPool3DAPIStatic):
    def setKernelSize(self):
        self.kernel_sizes = [5, 5, 5]

    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 6, 9, 6, 3))


class TestInputStatic(TestMaxPool3DAPIStatic):
    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((2, 6, 7, 9, 3))
        dropout = paddle.nn.Dropout(0.8)
        self.dense_x = dropout(self.dense_x)


if __name__ == "__main__":
    unittest.main()
