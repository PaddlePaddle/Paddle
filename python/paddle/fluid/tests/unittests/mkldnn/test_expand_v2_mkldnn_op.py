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
        self.attrs = {'shape': self.shape, 'use_mkldnn': True}
        self.set_inputs()
        self.set_additional_inputs()
        output = np.tile(self.x, self.expand_times)
        self.outputs = {'Out': output}

    def set_inputs(self):
        self.inputs = {'X': self.x}

    def set_additional_inputs(self):
        pass

    def init_data(self):
        self.ori_shape = [1, 1, 1, 140]
        self.shape = [2, 3, 4, 140]
        self.expand_times = [2, 3, 4, 1]

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


class TestExpandV2ExpandShapesTensor1OneDNNOp(TestExpandV2OneDNNOp):

    def init_data(self):
        self.ori_shape = [100, 1]
        self.expand_times = [1, 2]
        self.expand_shape = [100, 2]
        self.shape = [100, 2]

    def calc_expand_shapes_tensor(self):
        self.expand_shapes_tensor = []
        for index, ele in enumerate(self.expand_shape):
            self.expand_shapes_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

    def set_additional_inputs(self):
        self.calc_expand_shapes_tensor()
        self.inputs['expand_shapes_tensor'] = self.expand_shapes_tensor


class TestExpandV2ExpandShapesTensor2OneDNNOp(
        TestExpandV2ExpandShapesTensor1OneDNNOp):

    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.expand_shape = [12, 14]
        self.shape = [12, -1]


class TestExpandV2ShapesTensorOneDNNOp(TestExpandV2OneDNNOp):

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2, 1]
        self.expand_shape = [2, 100]
        self.shape = [2, 100]

    def set_additional_inputs(self):
        self.inputs['Shape'] = np.array(self.expand_shape).astype("int32")


#   BF16 TESTS
def create_expand_v2_bf16_test_class(parent):

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestExpandV2BF16OneDNNOp(parent):

        def set_inputs(self):
            self.attrs['mkldnn_data_type'] = 'bfloat16'
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
create_expand_v2_bf16_test_class(TestExpandV2ExpandShapesTensor1OneDNNOp)
create_expand_v2_bf16_test_class(TestExpandV2ExpandShapesTensor2OneDNNOp)
create_expand_v2_bf16_test_class(TestExpandV2ShapesTensorOneDNNOp)

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
