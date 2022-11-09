#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
from xpu.get_test_cover_info import (
    create_test_class,
    get_xpu_op_support_types,
    XPUOpTestWrapper,
)


class XPUTestClipByNormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'clip_by_norm'
        self.use_dynamic_create_class = False

    class TestClipByNormOp(XPUOpTest):
        def setUp(self):
            self.op_type = "clip_by_norm"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.use_xpu = True
            self.max_relative_error = 0.006
            self.initTestCase()
            input = np.random.random(self.shape).astype(self.dtype)
            input[np.abs(input) < self.max_relative_error] = 0.5
            self.inputs = {
                'X': input,
            }
            self.attrs = {}
            self.attrs['max_norm'] = self.max_norm
            norm = np.sqrt(np.sum(np.square(input)))
            if norm > self.max_norm:
                output = self.max_norm * input / norm
            else:
                output = input
            self.outputs = {'Out': output}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                self.check_output_with_place(self.place)

        def initTestCase(self):
            self.shape = (100,)
            self.max_norm = 1.0

    class TestCase1(TestClipByNormOp):
        def initTestCase(self):
            self.shape = (100,)
            self.max_norm = 1e20

    class TestCase2(TestClipByNormOp):
        def initTestCase(self):
            self.shape = (16, 16)
            self.max_norm = 0.1

    class TestCase3(TestClipByNormOp):
        def initTestCase(self):
            self.shape = (4, 8, 16)
            self.max_norm = 1.0


support_types = get_xpu_op_support_types('clip_by_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestClipByNormOp, stype)


if __name__ == "__main__":
    unittest.main()
