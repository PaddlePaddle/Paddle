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
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestAccuracyOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'accuracy'
        self.use_dynamic_create_class = False

    class TestXPUAccuracyOp(XPUOpTest):

        def setUp(self):
            self.op_type = "accuracy"
            self.init_dtype()
            n = 8192
            infer = np.random.random((n, 1)).astype(self.dtype)
            indices = np.random.randint(0, 2, (n, 1)).astype('int64')
            label = np.random.randint(0, 2, (n, 1)).astype('int64')
            self.inputs = {'Out': infer, 'Indices': indices, "Label": label}
            num_correct = 0
            for rowid in range(n):
                for ele in indices[rowid]:
                    if ele == label[rowid]:
                        num_correct += 1
                        break
            self.outputs = {
                'Accuracy':
                np.array([num_correct / float(n)]).astype(self.dtype),
                'Correct': np.array([num_correct]).astype("int32"),
                'Total': np.array([n]).astype("int32")
            }
            self.attrs = {'use_xpu': True}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)


support_types = get_xpu_op_support_types('accuracy')
for stype in support_types:
    create_test_class(globals(), XPUTestAccuracyOp, stype)

if __name__ == '__main__':
    unittest.main()
