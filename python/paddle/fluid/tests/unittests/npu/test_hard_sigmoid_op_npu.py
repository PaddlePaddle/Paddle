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
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.), 0.).astype(x.dtype)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUHardSigmoid(OpTest):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.set_npu()
        self.init_dtype()
        self.set_attrs()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        lower_threshold = -self.offset / self.slope
        upper_threshold = (1. - self.offset) / self.slope

        # Same reason as TestAbs
        delta = 0.005
        x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
        x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

        out = ref_hardsigmoid(x, self.slope, self.offset)

        self.attrs = {'slope': self.slope, 'offset': self.offset}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['X'], 'Out')

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_attrs(self):
        self.slope = 0.166666666666667
        self.offset = 0.5


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUHardSigmoid2(TestNPUHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.5


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUHardSigmoid3(TestNPUHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.4


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUHardSigmoidFp16(TestNPUHardSigmoid):
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
