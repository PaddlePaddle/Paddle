# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class TensorTypeTest(unittest.TestCase):
    def func_type_totensor(self):
        paddle.disable_static()
        inx = np.array([1, 2])
        tensorx = paddle.to_tensor(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual((typex_str == expectx), True)

    def test_type_totensor(self):
        with _test_eager_guard():
            self.func_type_totensor()
        self.func_type_totensor()

    def func_type_Tensor(self):
        paddle.disable_static()
        inx = np.array([1, 2])
        tensorx = paddle.Tensor(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual((typex_str == expectx), True)

        tensorx = paddle.tensor.logic.Tensor(inx)
        typex_str = str(type(tensorx))

        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual((typex_str == expectx), True)

    def test_type_Tensor(self):
        with _test_eager_guard():
            self.func_type_Tensor()
        self.func_type_Tensor()

    def func_type_core(self):
        paddle.disable_static()
        inx = np.array([1, 2])
        tensorx = core.VarBase(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual((typex_str == expectx), True)

        tensorx = paddle.framework.VarBase(inx)
        typex_str = str(type(tensorx))
        expectx = "<class 'paddle.Tensor'>"
        self.assertEqual((typex_str == expectx), True)

    def test_type_core(self):
        with _test_eager_guard():
            pass
        self.func_type_core()


if __name__ == '__main__':
    unittest.main()
