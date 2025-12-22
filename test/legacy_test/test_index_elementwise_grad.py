# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestIndexElementwiseGrad(unittest.TestCase):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"

    def setUp(self):
        self.init()

        if self.dtype in ["float32", "float64"]:
            self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        elif self.dtype in ["int32", "int8", "int64", "int16", "uint8"]:
            self.x_np = np.random.randint(
                100, size=self.x_shape, dtype=self.dtype
            )
        elif self.dtype == "float16":
            self.x_np = np.random.random(self.x_shape).astype("float16")

        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )

    def test_grad(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        index = paddle.to_tensor(self.index_np).astype('bool')

        out = x[index]
        out_grad = paddle.ones_like(out)
        out.backward(out_grad)
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        x_grad_np = x.grad.numpy()
        expanded_index = np.expand_dims(
            self.index_np, axis=tuple(range(self.k, self.dim))
        )
        expanded_index = np.broadcast_to(expanded_index, self.x_shape)
        expected_grad = np.where(expanded_index, 1.0, 0.0).astype(self.dtype)

        atol = 1e-5 if self.dtype in ["float32", "float64"] else 1e-3
        rtol = 1e-5 if self.dtype in ["float32", "float64"] else 1e-3

        np.testing.assert_allclose(
            x_grad_np, expected_grad, atol=atol, rtol=rtol
        )

        paddle.enable_static()


class TestIndexElementwiseGrad3D(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad4D_k2(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad4D_k3(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k2(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k3(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k4(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 4
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGradFloat64(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float64"


class TestIndexElementwiseGradFloat16(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float16"

    def setUp(self):
        self.init()
        self.x_np = np.random.random(self.x_shape).astype("float16")
        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )


class TestIndexElementwiseGradWithCustomOutGrad(unittest.TestCase):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"

    def setUp(self):
        self.init()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )

    def test_custom_out_grad(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        index = paddle.to_tensor(self.index_np).astype('bool')

        out = x[index]
        custom_grad = paddle.randn_like(out)
        out.backward(custom_grad)

        self.assertEqual(x.grad.shape, x.shape)
        paddle.enable_static()


class TestIndexElementwiseGradZeroIndex(unittest.TestCase):
    def test_zero_index(self):
        paddle.disable_static()

        x = paddle.randn([4, 5, 6], dtype='float32')
        x.stop_gradient = False
        index = paddle.zeros([4, 5], dtype='bool')
        out = x[index]
        self.assertEqual(out.numel(), 0)
        if out.numel() > 0:
            out.backward(paddle.ones_like(out))
            np.testing.assert_allclose(
                x.grad.numpy(), np.zeros_like(x.numpy()), atol=1e-5
            )

        paddle.enable_static()


class TestIndexElementwiseGradAllIndex(unittest.TestCase):
    def test_all_index(self):
        paddle.disable_static()

        x_np = np.random.random([4, 5, 6]).astype('float32')
        x = paddle.to_tensor(x_np, stop_gradient=False)
        index = paddle.ones([4, 5], dtype='bool')
        out = x[index]
        out.backward(paddle.ones_like(out))
        expected_grad = np.ones_like(x_np)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, atol=1e-5, rtol=1e-5
        )

        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
