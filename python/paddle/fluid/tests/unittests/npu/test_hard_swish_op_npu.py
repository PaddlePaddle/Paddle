#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F


def ref_hard_swish_grad(x, threshold=6.0, scale=6.0, offset=3.0):
    dout = np.full_like(x, fill_value=1. / x.size)
    tmp = ((x + offset) < threshold).astype(x.dtype)
    dx = dout * (((x + offset) > 0).astype(x.dtype) *
                 (2 * x + offset) * tmp / scale + 1.0 - tmp)
    return dx


class TestHardSwishNPU(OpTest):
    def setUp(self):
        paddle.enable_static()

        self.set_npu()
        self.op_type = "hard_swish"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()

        x = np.random.uniform(-6, 6, [10, 12]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        #the same with TestAbs
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = (x * (np.minimum(np.maximum(x + offset, 0.), threshold) /
                    scale)).astype(self.dtype)
        self.x_grad = ref_hard_swish_grad(x, threshold, scale, offset)

        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # There is a problem that precision of grad result using float32
        # can't satisfy the default precision requirement 
        # when compared with numeric_grads, but the results on 
        # NPU and CPU are same (verified in TestHardSwishNPUWithCPU)
        self.check_grad_with_place(
            self.place, ['X'], 'Out', user_defined_grads=[self.x_grad])


class TestHardSwishNPUFp16(TestHardSwishNPU):
    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_dtype(self):
        self.dtype = np.float16


# test the result of hard_swish and hard_swish_grad on CPU and NPU
class TestHardSwishNPUWithCPU(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

        self.place = paddle.NPUPlace(0)
        self.dtype = np.float32

        self.x = np.random.uniform(-6, 10, [8, 15]).astype(self.dtype)

        paddle.set_device('cpu')

        data = paddle.to_tensor(self.x, stop_gradient=False)
        y = F.hardswish(data)
        y.sum().backward()

        self.out_g = data.grad
        self.out_y = y

    def test_check_output_and_grad_npu(self):
        paddle.set_device('npu')

        data = paddle.to_tensor(self.x, stop_gradient=False)
        y = F.hardswish(data)
        y.sum().backward()

        self.assertTrue(
            np.allclose(self.out_y.numpy(), y.numpy()),
            "Output of NPU HardSwish forward has diff at " + str(self.place) +
            "\nExpect " + str(self.out_y) + "\n" + "But Got" + str(y) +
            " in class " + self.__class__.__name__ + ".")
        self.assertTrue(
            np.allclose(self.out_g.numpy(), data.grad.numpy()),
            "Output of NPU HardSwish backward has diff at " + str(self.place) +
            "\nExpect " + str(self.out_g) + "\n" + "But Got" + str(data.grad) +
            " in class " + self.__class__.__name__ + ".")


if __name__ == '__main__':
    unittest.main()
