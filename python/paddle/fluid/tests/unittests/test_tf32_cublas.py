#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestTF32Switch(unittest.TestCase):
    def test_on_off(self):
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            ctx = core.CUDADeviceContext(place)
            self.assertTrue(ctx.get_cublas_switch())  # default
            ctx.set_cublas_switch(0)
            self.assertFalse(ctx.get_cublas_switch())  # turn off
            ctx.set_cublas_switch(1)
            self.assertTrue(ctx.get_cublas_switch())  # turn on

            ctx.set_cublas_switch(1)  # restore the switch
        else:
            pass


class TestTF32OnMatmul(unittest.TestCase):
    def test_dygraph_without_out(self):
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            tmp_ctx = core.CUDADeviceContext(place)
            print(tmp_ctx.get_cublas_switch())
            tmp_ctx.set_cublas_switch(0)  # turn off
            print(tmp_ctx.get_cublas_switch())
            with fluid.dygraph.guard(place):
                input_array1 = np.random.rand(4, 12, 64, 88).astype("float32")
                input_array2 = np.random.rand(4, 12, 88, 512).astype("float32")
                data1 = paddle.to_tensor(input_array1)
                data2 = paddle.to_tensor(input_array2)
                out = paddle.matmul(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
            self.assertTrue(np.allclose(expected_result, out.numpy(), 1e-03))
            tmp_ctx.set_cublas_switch(1)  # restore the switch
        else:
            pass


if __name__ == '__main__':
    unittest.main()
