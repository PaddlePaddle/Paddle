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
import numpy as np
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle._legacy_C_ops as ops


class TestGetFloatStatusOp(unittest.TestCase):

    def setUp(self):
        device = paddle.set_device('npu')

    def run_prog(self, a, b):
        a = paddle.to_tensor(a)
        b = paddle.to_tensor(b)

        flag = ops.alloc_float_status()
        ops.clear_float_status(flag, flag)

        out = a / b
        ops.get_float_status(flag, flag)
        return out.numpy(), flag.numpy()

    def test_contains_nan(self):
        a = np.zeros((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')

        out, flag = self.run_prog(a, b)
        print(out, flag)
        self.assertGreaterEqual(np.sum(flag), 1.0)

    def test_contains_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')

        out, flag = self.run_prog(a, b)
        print(out, flag)
        self.assertGreaterEqual(np.sum(flag), 1.0)

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.ones((32, 32)).astype('float32')

        out, flag = self.run_prog(a, b)
        print(out, flag)
        self.assertLess(np.sum(flag), 1.0)


class TestClearFloatStatusOp(unittest.TestCase):

    def setUp(self):
        device = paddle.set_device('npu')

    def run_prog(self, a, b):
        a = paddle.to_tensor(a)
        b = paddle.to_tensor(b)

        flag = ops.alloc_float_status()
        ops.clear_float_status(flag, flag)

        out = a / b
        ops.get_float_status(flag, flag)

        ops.clear_float_status(flag, flag)
        out = a + b
        ops.get_float_status(flag, flag)
        return out.numpy(), flag.numpy()

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')

        out, flag = self.run_prog(a, b)
        print(out, flag)
        self.assertLess(np.sum(flag), 1.0)

    def test_fp16_overflow(self):
        a = np.ones((32, 32)).astype('float16')
        b = np.zeros((32, 32)).astype('float16')
        a[0][0] = 50000
        b[0][0] = 50000

        out, flag = self.run_prog(a, b)
        print(out, flag)
        self.assertGreaterEqual(np.sum(flag), 1.0)


if __name__ == '__main__':
    unittest.main()
