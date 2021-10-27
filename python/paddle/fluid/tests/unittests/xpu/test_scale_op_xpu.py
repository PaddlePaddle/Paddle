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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")
from op_test_xpu import XPUOpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle
from paddle.static import Program, program_guard


class TestXPUScaleOp(XPUOpTest):
    def setUp(self):
        self.op_type = "scale"
        self.init_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3, 'use_xpu': True}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def init_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


# class TestXPUScaleOpInt64(TestXPUScaleOp):
#     def init_type(self):
#         self.dtype = np.int64


class TestScaleFp16Op(TestXPUScaleOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.XPUPlace(0)
        self.check_output_with_place(place, atol=0.002)

    def test_check_grad(self):
        place = core.XPUPlace(0)
        self.check_grad_with_place(place, ["X"], "Out", max_relative_error=0.05)


class TestScaleApiStatic(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        main_prog = Program()
        with program_guard(main_prog, Program()):
            x = paddle.static.data(name="x", shape=[2, 25], dtype="float32")
            out = self._executed_api(x, scale=2.0, bias=3.0)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        out = exe.run(main_prog, feed={"x": input}, fetch_list=[out])
        self.assertEqual(np.array_equal(out[0], input * 2.0 + 3.0), True)


class TestScaleInplaceApiStatic(TestScaleApiStatic):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


class TestScaleApiDygraph(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.disable_static()
        input = np.random.random([2, 25]).astype("float32")
        x = paddle.to_tensor(input)
        out = self._executed_api(x, scale=2.0, bias=3.0)
        self.assertEqual(np.array_equal(out.numpy(), input * 2.0 + 3.0), True)
        paddle.enable_static()


class TestScaleInplaceApiDygraph(TestScaleApiDygraph):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


if __name__ == "__main__":
    unittest.main()
