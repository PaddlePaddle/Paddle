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

import unittest
import sys

sys.path.append("..")
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestFlattenOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'flatten'
        self.use_dynamic_create_class = False

    class TestFlattenOp(XPUOpTest):

        def setUp(self):
            self.op_type = "flatten"
            self.use_xpu = True
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            self.dtype = self.in_type
            self.inputs = {
                "X": np.random.random(self.in_shape).astype(self.dtype)
            }
            self.init_attrs()
            self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape)}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ["X"], "Out")

        def init_test_case(self):
            self.in_shape = (3, 2, 2, 10)
            self.axis = 1
            self.new_shape = (3, 40)

        def init_attrs(self):
            self.attrs = {"axis": self.axis}

    class TestFlattenOp1(TestFlattenOp):

        def init_test_case(self):
            self.in_shape = (3, 2, 2, 10)
            self.axis = 0
            self.new_shape = (1, 120)

    class TestFlattenOpWithDefaultAxis(TestFlattenOp):

        def init_test_case(self):
            self.in_shape = (10, 2, 2, 3)
            self.new_shape = (10, 12)

        def init_attrs(self):
            self.attrs = {}

    class TestFlattenOpSixDims(TestFlattenOp):

        def init_test_case(self):
            self.in_shape = (3, 2, 3, 2, 4, 4)
            self.axis = 4
            self.new_shape = (36, 16)


support_types = get_xpu_op_support_types('flatten')
for stype in support_types:
    create_test_class(globals(), XPUTestFlattenOp, stype)

if __name__ == "__main__":
    unittest.main()
