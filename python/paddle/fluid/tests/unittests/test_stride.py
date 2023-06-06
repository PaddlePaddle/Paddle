# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestStride(unittest.TestCase):
    def call_transpose(self):
        x_np = np.random.random(size=[2, 3, 4]).astype('float32')
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)
        self.assertTrue(np.allclose(x_transposed1.numpy(), x_np_transposed1))
        self.assertFalse(x_transposed1.is_contiguous("NCHW"))
        self.assertTrue(x._is_shared_buffer_with(x_transposed1))

        x_c = x_transposed1.contiguous()
        self.assertTrue(np.allclose(x_c.numpy(), x_np_transposed1))
        self.assertFalse(x_c._is_shared_buffer_with(x_transposed1))

        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        self.assertTrue(np.allclose(x_transposed2.numpy(), x_np_transposed2))
        self.assertFalse(x_transposed2.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(x_transposed2))

        y = x_transposed2 + 2
        y_np = x_np_transposed2 + 2
        self.assertTrue(np.allclose(y.numpy(), y_np))
        self.assertTrue(y.is_contiguous())
        self.assertFalse(x._is_shared_buffer_with(y))

    def call_diagonal(self):
        x_np = np.random.random(size=[2, 3, 4]).astype('float32')
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        out = paddle.diagonal(x)
        out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
        out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
        out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)

        np_out = np.diagonal(x_np)
        np_out2 = np.diagonal(x_np, offset=0, axis1=2, axis2=1)
        np_out3 = np.diagonal(x_np, offset=1, axis1=0, axis2=1)
        np_out4 = np.diagonal(x_np, offset=0, axis1=1, axis2=2)

        self.assertTrue(np.allclose(out.numpy(), np_out))
        self.assertTrue(np.allclose(out2.numpy(), np_out2))
        self.assertTrue(np.allclose(out3.numpy(), np_out3))
        self.assertTrue(np.allclose(out4.numpy(), np_out4))

        self.assertFalse(out.is_contiguous("NCHW"))
        self.assertFalse(out2.is_contiguous("NCHW"))
        self.assertFalse(out3.is_contiguous("NCHW"))
        self.assertFalse(out4.is_contiguous("NCHW"))

        self.assertTrue(x._is_shared_buffer_with(out))
        self.assertTrue(x._is_shared_buffer_with(out2))
        self.assertTrue(x._is_shared_buffer_with(out3))
        self.assertTrue(x._is_shared_buffer_with(out4))

        out_c = out.contiguous()
        out2_c = out2.contiguous()
        out3_c = out3.contiguous()
        out4_c = out4.contiguous()

        self.assertTrue(np.allclose(out_c.numpy(), np_out))
        self.assertTrue(np.allclose(out2_c.numpy(), np_out2))
        self.assertTrue(np.allclose(out3_c.numpy(), np_out3))
        self.assertTrue(np.allclose(out4_c.numpy(), np_out4))

        self.assertFalse(out_c._is_shared_buffer_with(out))
        self.assertFalse(out2_c._is_shared_buffer_with(out2))
        self.assertFalse(out3_c._is_shared_buffer_with(out3))
        self.assertFalse(out4_c._is_shared_buffer_with(out4))

    def call_stride(self):
        self.call_transpose()
        self.call_diagonal()


class TestStrideCPU(TestStride):
    def test_stride_cpu(self):
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('cpu')
        self.call_stride()
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})


@unittest.skipIf(
    not paddle.fluid.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestStrideGPU(TestStride):
    def test_stride_gpu(self):
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('gpu')
        self.call_stride()
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})


if __name__ == '__main__':
    unittest.main()
