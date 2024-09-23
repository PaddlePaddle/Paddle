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
from paddle.incubate.nn import FusedLinear
from paddle.incubate.nn.functional import fused_linear, fused_matmul_bias


def is_fused_matmul_bias_supported():
    return hasattr(core.eager.ops.legacy, 'fused_gemm_epilogue')


def matmul(x, y, bias, trans_x, trans_y):
    x = np.array(x)
    if trans_x:
        x = np.ascontiguousarray(np.transpose(x))
    if trans_y:
        y = np.ascontiguousarray(np.transpose(y))
    z = np.matmul(x, y)
    if bias is None:
        return z
    else:
        return z + bias


def matmul_grad(x, y, bias, dz, trans_x, trans_y):
    if trans_x:
        if trans_y:
            dx = matmul(y, dz, None, True, True)
            dy = matmul(dz, x, None, True, True)
        else:
            dx = matmul(y, dz, None, False, True)
            dy = matmul(x, dz, None, False, False)
    else:
        if trans_y:
            dx = matmul(dz, y, None, False, False)
            dy = matmul(dz, x, None, True, False)
        else:
            dx = matmul(dz, y, None, False, True)
            dy = matmul(x, dz, None, True, False)
    if bias is None:
        dbias = None
    else:
        dbias = np.sum(dz, axis=0, keepdims=False)
    return dx, dy, dbias


@unittest.skipIf(
    not is_fused_matmul_bias_supported(),
    "fused_gemm_epilogue is only supported when CUDA version >= 11.6",
)
class TestFusedMatmulBias(unittest.TestCase):
    def setUp(self):
        paddle.set_device('gpu')

    def rand_data(self, shape, dtype):
        return np.random.randint(low=-20, high=20, size=shape).astype(dtype)

    def rand_test_base(self, m, n, k, trans_x, trans_y, need_bias, dtype, seed):
        np.random.seed(seed)
        x_shape = [k, m] if trans_x else [m, k]
        y_shape = [n, k] if trans_y else [k, n]
        bias_shape = [n]

        x_np = self.rand_data(x_shape, dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False

        y_np = self.rand_data(y_shape, dtype)
        y = paddle.to_tensor(y_np)
        y.stop_gradient = False

        if need_bias:
            bias_np = self.rand_data(bias_shape, dtype)
            bias = paddle.to_tensor(bias_np)
            bias.stop_gradient = False
        else:
            bias_np = None
            bias = None

        z = fused_matmul_bias(x, y, bias, trans_x, trans_y)
        z_np = matmul(x_np, y_np, bias_np, trans_x, trans_y)
        np.testing.assert_array_equal(z.numpy(), z_np)

        z_grad_np = self.rand_data(z_np.shape, dtype)
        paddle.autograd.backward(z, grad_tensors=[paddle.to_tensor(z_grad_np)])

        x_grad_np, y_grad_np, bias_grad_np = matmul_grad(
            x_np, y_np, bias_np, z_grad_np, trans_x, trans_y
        )
        np.testing.assert_array_equal(x.grad.numpy(), x_grad_np)
        self.assertEqual(y_grad_np.shape, y_np.shape)
        np.testing.assert_array_equal(y.grad.numpy(), y_grad_np)

        if need_bias:
            np.testing.assert_array_equal(bias.grad.numpy(), bias_grad_np)
        else:
            self.assertIsNone(bias_grad_np)

    def rand_test(self, m, n, k, dtype):
        seed = int(np.random.randint(low=0, high=1000, size=[1]))
        for trans_x in [False, True]:
            for trans_y in [False, True]:
                for need_bias in [False, True]:
                    self.rand_test_base(
                        m, n, k, trans_x, trans_y, need_bias, dtype, seed
                    )

    def test_fp32(self):
        self.rand_test(30, 40, 50, np.float32)

    def test_fp16(self):
        self.rand_test(4, 5, 7, np.float16)


@unittest.skipIf(
    not is_fused_matmul_bias_supported(),
    "fused_gemm_epilogue is only supported when CUDA version >= 11.6",
)
class TestFusedLinear(unittest.TestCase):
    def check_fused_linear(self, transpose):
        x = paddle.randn([30, 40])
        linear = FusedLinear(40, 50, transpose_weight=transpose)
        y1 = linear(x)
        y2 = fused_linear(x, linear.weight, linear.bias, transpose)
        np.testing.assert_array_equal(y1.numpy(), y2.numpy())

    def test_non_transpose(self):
        self.check_fused_linear(False)

    def test_transpose(self):
        self.check_fused_linear(True)


@unittest.skipIf(
    not is_fused_matmul_bias_supported(),
    "fused_gemm_epilogue is only supported when CUDA version >= 11.6",
)
class TestStaticGraph(unittest.TestCase):

    def test_static_graph(self):
        paddle.enable_static()
        x = paddle.static.data(name='x', dtype='float32', shape=[-1, 100])
        linear = FusedLinear(100, 300)
        y = linear(x)
        self.assertEqual(list(y.shape), [-1, 300])
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
