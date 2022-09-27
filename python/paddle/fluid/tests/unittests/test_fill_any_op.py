#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _test_eager_guard, in_dygraph_mode
import unittest
import numpy as np
from op_test import OpTest
from paddle.tensor.manipulation import fill_


class TestFillAnyOp(OpTest):

    def setUp(self):
        self.op_type = "fill_any"
        self.dtype = 'float64'
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((20, 30)).astype(self.dtype)}
        self.attrs = {
            'value_float': float(self.value),
            'value_int': int(self.value)
        }
        self.outputs = {
            'Out':
            self.value * np.ones_like(self.inputs["X"]).astype(self.dtype)
        }

    def init(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestFillAnyOpFloat32(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 0.0


class TestFillAnyOpFloat16(TestFillAnyOp):

    def init(self):
        self.dtype = np.float16


class TestFillAnyOpvalue1(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 111111555


class TestFillAnyOpvalue2(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 11111.1111


class TestFillAnyInplace(unittest.TestCase):

    def test_fill_any_version(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 1)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 2)

            var.fill_(0)
            self.assertEqual(var.inplace_version, 3)

    def test_fill_any_eqaul(self):
        with paddle.fluid.dygraph.guard():
            tensor = paddle.to_tensor(
                np.random.random((20, 30)).astype(np.float32))
            target = tensor.numpy()
            target[...] = 1

            tensor.fill_(1)
            self.assertEqual((tensor.numpy() == target).all().item(), True)

    def test_backward(self):
        with paddle.fluid.dygraph.guard():
            x = paddle.full([10, 10], -1., dtype='float32')
            x.stop_gradient = False
            y = 2 * x
            y.fill_(1)
            y.backward()
            np.testing.assert_array_equal(x.grad.numpy(), np.zeros([10, 10]))


if __name__ == "__main__":
    unittest.main()
