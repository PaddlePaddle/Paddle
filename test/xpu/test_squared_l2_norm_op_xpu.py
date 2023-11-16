# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


paddle.seed(10)


class XPUTestSquaredL2NormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'squared_l2_norm'
        self.use_dynamic_create_class = False

    class TestSquaredL2NormOp(XPUOpTest):
        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'squared_l2_norm'

            # no grad implement at this time
            self.__class__.no_need_check_grad = True

        def setUp(self):
            self.init()
            self.use_mkldnn = False
            self.max_relative_error = 0.05
            self.set_inputs()
            self.inputs = {'X': self.x}
            self.outputs = {
                'Out': np.array([np.square(np.linalg.norm(self.x))])
            }

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def set_inputs(self):
            self.x = np.random.uniform(-1, 1, (13, 19)).astype(self.in_type)
            self.x[np.abs(self.x) < self.max_relative_error] = 0.1

    class TestSquaredL2NormOp_1(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.2, 0.2, (8, 128, 24, 6)).astype(
                self.in_type
            )
            self.x[np.abs(self.x) < self.max_relative_error] = 0.05

    class TestSquaredL2NormOp_2(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (2, 128, 512)).astype(
                self.in_type
            )
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01


support_types = get_xpu_op_support_types('squared_l2_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestSquaredL2NormOp, stype)

if __name__ == "__main__":
    unittest.main()
