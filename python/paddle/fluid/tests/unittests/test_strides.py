"""
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
"""

import sys
import unittest

import numpy as np

import paddle

"""
the default test dtype.
"""
dtype = None


class TestStrides(unittest.TestCase):
    """
    TestStrides - the unittest for the strides test.
    """

    def call_strides(self, dtype="int32"):
        """
        Summary:
          the unittest routine for the strides test on current device and specified dtype.
        Args:
          dtype (string): the specified dtype for the current testcase.
        """

        if dtype in ["int8", "int16", "int32", "int64"]:
            x_np = np.arange(0, 24, 1).astype(dtype).reshape([2, 3, 4])
        elif dtype in ["float16", "float32", "float64"]:
            x_np = np.random.random(size=[2, 3, 4]).astype(dtype)
        else:
            print("invalid dtype", dtype)
            return

        print("CURRENT_DTYPE:", dtype)
        x = paddle.to_tensor(x_np)
        print("x", x)
        print("x.numpy", x.numpy())
        print("x_np", x_np)
        print("x.strides", x.get_strides())

        self.assertTrue(np.allclose(x.numpy(), x_np))
        self.assertTrue(x.is_contiguous())
        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)

        print("x_transposed1", x_transposed1)
        print("x_transposed1.numpy()", x_transposed1.numpy())
        print("x_np_transposed1", x_np_transposed1)
        print("x_transposed1.strides", x_transposed1.strides)
        print("x_transposed1.get_strides()", x_transposed1.get_strides())

        self.assertTrue(np.allclose(x_transposed1.numpy(), x_np_transposed1))
        print("x_transposed1.is_dense()", x_transposed1.is_dense())
        print("x_transposed1.is_contiguous()", x_transposed1.is_contiguous())
        print(
            "x_transposed1.is_contiguous(\"NCHW\")",
            x_transposed1.is_contiguous("NCHW"),
        )

        self.assertFalse(x_transposed1.is_contiguous())
        self.assertFalse(x_transposed1.is_contiguous("NCHW"))

        self.assertTrue(x._is_shared_buffer_with(x_transposed1))

        x_c = x_transposed1.contiguous()
        print("x_c", x_c)
        print("x_c.strides", x_c.strides)
        print("x_c.get_strides()", x_c.get_strides())
        print("x_c.is_dense()", x_c.is_dense())
        print("x_c.is_contiguous()", x_c.is_contiguous())
        print("x_c.is_contiguous(\"NCHW\")", x_c.is_contiguous("NCHW"))

        self.assertTrue(np.allclose(x_c.numpy(), x_np_transposed1))
        self.assertFalse(x_c._is_shared_buffer_with(x_transposed1))

        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        self.assertTrue(np.allclose(x_transposed2.numpy(), x_np_transposed2))
        self.assertFalse(x_transposed2.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(x_transposed2))

        print("x_transposed2", x_transposed2)
        print("x_np_transposed2", x_np_transposed2)

        y = x_transposed2 + 2
        print("y", y)
        y_np = x_np_transposed2 + 2
        print("y_np", y_np)
        print("y.numpy()", y.numpy())
        self.assertTrue(np.allclose(y.numpy(), y_np))
        self.assertTrue(y.is_contiguous())
        self.assertFalse(x._is_shared_buffer_with(y))


class TestStridesCPU(TestStrides):
    """
    TestStridesCPU - the unittest for the strides test on cpu device.
    """

    def test_strides_cpu(self):
        """
        the unittest routine for the strides test on cpu device.
        """
        print("_____CPU______BEGIN____")
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('cpu')
        global dtype
        if dtype is None:
            self.call_strides("int8")
            self.call_strides("int16")
            self.call_strides("int32")
            self.call_strides("int64")
            self.call_strides("float32")
            self.call_strides("float64")
        else:
            self.call_strides(dtype)
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})
        print("_____CPU______END____")


@unittest.skipIf(
    not paddle.fluid.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestStridesGPU(TestStrides):
    """
    the unittest for the strides test on gpu device.
    """

    def test_strides_gpu(self):
        """
        the unittest routine for the strides test on gpu device.
        """
        print("_____GPU______BEGIN____")
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": True})
        paddle.set_device('gpu')
        global dtype
        if dtype is None:
            self.call_strides("int8")
            self.call_strides("int16")
            self.call_strides("int32")
            self.call_strides("int64")
            self.call_strides("float32")
            self.call_strides("float64")
            self.call_strides("float16")
        else:
            self.call_strides(dtype)
        paddle.fluid.set_flags({"FLAGS_use_stride_kernel": False})
        print("_____GPU______END____")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dtype = sys.argv[1]
        sys.argv.remove(dtype)
    unittest.main()
