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

from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16


@OpTestTool.skip_if(core.is_compiled_with_cuda(),
                    "CUDA required dygraph so oneDNN UT must be skipped")
class TestExpandV2OneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.set_inputs()
        self.attrs = {'shape': self.shape, 'use_mkldnn': True}
        output = np.tile(self.x, self.expand_times)
        self.outputs = {'Out': output}

    def set_inputs(self):
        self.inputs = {'X': self.x}

    def init_data(self):
        self.ori_shape = [1, 140]
        self.shape = [12, 140]
        self.expand_times = [12, 1]

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        self.check_grad_with_place(core.CPUPlace(), ["X"], "Out")


class TestExpandV2ExpandDimOneDNNOp(TestExpandV2OneDNNOp):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [2, 120]
        self.expand_times = [2, 1]


class TestExpandV2CopyScenarioOneDNNOp(TestExpandV2OneDNNOp):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2CopyScenarioShapeNotGivenOneDNNOp(TestExpandV2OneDNNOp):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (-1, -1, -1, -1)
        self.expand_times = (1, 1, 1, 1)


#   BF16 TESTS
def create_expand_v2_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestExpandV2BF16OneDNNOp(parent):
        def set_inputs(self):
            self.inputs = {"X": convert_float_to_uint16(self.x)}

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = self.dout.copy()

            for i in range(len(self.shape)):
                if self.expand_times[i] != 1:
                    self.dx = np.sum(self.dx, axis=i, keepdims=True)

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X"],
                "Out",
                user_defined_grads=[convert_float_to_uint16(self.dx)],
                user_defined_grad_outputs=[self.dout])

    cls_name = "{0}_{1}".format(parent.__name__, "Expand_v2_BF16")
    TestExpandV2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestExpandV2BF16OneDNNOp


create_expand_v2_bf16_test_class(TestExpandV2OneDNNOp)
create_expand_v2_bf16_test_class(TestExpandV2ExpandDimOneDNNOp)
create_expand_v2_bf16_test_class(TestExpandV2CopyScenarioOneDNNOp)
create_expand_v2_bf16_test_class(TestExpandV2CopyScenarioShapeNotGivenOneDNNOp)

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
