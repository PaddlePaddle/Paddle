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

paddle.enable_static()
SEED = 2021

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestHardSwishNPU(OpTest):
    def setUp(self):
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
        out = (x * np.minimum(np.maximum(x + offset, 0.), threshold) /
            scale).astype(x.dtype)

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
        self.check_grad_with_place(
            self.place, ['X'], 'Out')


# @unittest.skipIf(not paddle.is_compiled_with_npu(),
#                  "core is not compiled with NPU")
# class TestHardSwishNPUFp16(TestHardSwishNPU):
#     def test_check_output(self):
#         self.check_output_with_place(self.place, atol=1e-3)

#     def init_dtype(self):
#         self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
