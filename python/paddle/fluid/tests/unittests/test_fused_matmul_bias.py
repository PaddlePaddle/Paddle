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

import paddle
import paddle.fluid.core as core
import unittest
import numpy as np
from paddle.incubate.nn.functional import fused_matmul_bias
from custom_setup_ops import custom_fused_dense


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.ops, 'fused_gemm_epilogue')
    else:
        return False


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


def _fused_matmul_bias(x, y, bias, trans_x, trans_y):
    if bias is None:
        return paddle.matmul(x, y, trans_x, trans_y)
    else:
        return custom_fused_dense(
            x=x,
            y=y,
            bias=bias,
            transx=trans_x,
            transy=trans_y,
            use_addto=False)


@unittest.skipIf(
    not is_fused_matmul_bias_supported(),
    "fused_gemm_epilogue is only supported when CUDA version >= 11.6")
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
        #z = paddle.matmul(x, y, trans_x, trans_y)
        #if bias is not None:
        #    z = z + bias
        z_np = matmul(x_np, y_np, bias_np, trans_x, trans_y)
        self.assertTrue(np.array_equal(z.numpy(), z_np))

        z_grad_np = self.rand_data(z_np.shape, dtype)
        paddle.autograd.backward(z, grad_tensors=[paddle.to_tensor(z_grad_np)])

        print(trans_x, trans_y, need_bias)
        x_grad_np, y_grad_np, bias_grad_np = matmul_grad(
            x_np, y_np, bias_np, z_grad_np, trans_x, trans_y)
        self.assertTrue(np.array_equal(x.grad.numpy(), x_grad_np))
        self.assertEqual(y_grad_np.shape, y_np.shape)
        print(y.grad.numpy(), y_grad_np)
        self.assertTrue(np.array_equal(y.grad.numpy(), y_grad_np))

        if need_bias:
            print(bias.grad.numpy(), bias_grad_np)
            if not np.array_equal(bias.grad.numpy(), bias_grad_np):
                print('Failure....')
            self.assertTrue(np.array_equal(bias.grad.numpy(), bias_grad_np))
        else:
            self.assertTrue(bias_grad_np is None)

    def rand_test(self, m, n, k, dtype):
        seed = int(np.random.randint(low=0, high=1000, size=[1]))
        for trans_x in [False]:
            for trans_y in [False, True]:
                for need_bias in [False, True]:
                    self.rand_test_base(m, n, k, trans_x, trans_y, need_bias,
                                        dtype, seed)

    def test_fp32(self):
        self.rand_test(3, 4, 5, np.float32)

    def _test_fp16(self):
        self.rand_test(4, 5, 7, np.float16)


if __name__ == "__main__":
    unittest.main()
