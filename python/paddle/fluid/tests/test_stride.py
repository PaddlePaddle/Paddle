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


class TestStrides(unittest.TestCase):
    def call_strides(self):
        x_np = np.random.random(size=[2, 2, 2]).astype('float32')
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))
        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)
        self.assertTrue(
            np.allclose(x_transposed1.contiguous().numpy(), x_np_transposed1)
        )
        self.assertFalse(x_transposed1.is_contiguous("NCHW"))
        self.assertTrue(x._is_shared_buffer_with(x_transposed1))

        x_c = x_transposed1.contiguous()
        self.assertTrue(np.allclose(x_c.numpy(), x_np_transposed1))
        self.assertFalse(x_c._is_shared_buffer_with(x_transposed1))

        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        self.assertTrue(
            np.allclose(x_transposed2.contiguous().numpy(), x_np_transposed2)
        )
        self.assertFalse(x_transposed2.is_contiguous())
        # self.assertTrue(x_transposed1._is_shared_buffer_with(x_transposed2))

        y = x_transposed2 + 2
        y_np = x_np_transposed2 + 2
        self.assertTrue(np.allclose(y.numpy(), y_np))
        self.assertTrue(y.is_contiguous())
        self.assertFalse(x._is_shared_buffer_with(y))

        # test squeeze&unsqueeze
        a = paddle.randn(shape=[1, 2, 4])
        b = a.squeeze([0])
        self.assertTrue(b.is_contiguous())
        self.assertTrue(a._is_shared_buffer_with(b))
        a.fill_(0)
        self.assertTrue(b.sum() == 0)
        c = a.unsqueeze([3])
        self.assertTrue(c._is_shared_buffer_with(b))
        self.assertTrue(c.is_contiguous())

        # test view&view_as
        a = paddle.randn(shape=[2, 8])
        b = a.view([4, 4])
        self.assertTrue(b.shape == [4, 4])
        self.assertTrue(a._is_shared_buffer_with(b))
        self.assertTrue(a.is_contiguous())
        c = b.view_as(a)
        self.assertTrue(c.shape == [2, 8])
        self.assertTrue(b._is_shared_buffer_with(c))

        # test flatten
        a = paddle.randn(shape=[2, 2])
        b = a.flatten()
        self.assertTrue(a._is_shared_buffer_with(b))


class TestStridesCPU(TestStrides):
    def test_strides_cpu(self):
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('cpu')
        self.call_strides()
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})


@unittest.skipIf(
    not paddle.fluid.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestStridesGPU(TestStrides):
    def test_strides_gpu(self):
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('gpu')
        self.call_strides()
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})


if __name__ == '__main__':
    unittest.main()
