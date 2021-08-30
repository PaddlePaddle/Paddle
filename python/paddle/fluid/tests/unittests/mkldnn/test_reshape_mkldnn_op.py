# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16


@OpTestTool.skip_if(core.is_compiled_with_cuda(),
                    "CUDA has to be skipped because it forces dygraph")
class TestReshape2OneDNNOp(OpTest):
    def setUp(self):
        self.init_data()
        self.set_op_type()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.set_inputs()
        self.set_additional_inputs()
        self.set_attrs()
        self.set_outputs()

    def set_op_type(self):
        self.op_type = "reshape2"

    def set_inputs(self):
        self.inputs = {"X": self.x}

    def set_additional_inputs(self):
        pass

    def set_attrs(self):
        self.attrs = {"shape": self.new_shape, 'use_mkldnn': True}

    def set_outputs(self):
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshape2OneDNNOpDimInfer1(TestReshape2OneDNNOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshape2OneDNNOpDimInfer2(TestReshape2OneDNNOp):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)

    def set_additional_inputs(self):
        self.inputs["Shape"] = np.array(self.actual_shape, dtype="int32")

    def set_outputs(self):
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.actual_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (6, 20)
        self.new_shape = (0, -1, 20)
        self.actual_shape = (2, 3, 20)


class TestReshape2OneDNNOp_attr_OnlyShape(TestReshape2OneDNNOp):
    def set_additional_inputs(self):
        self.inputs["Shape"] = np.array(self.new_shape, dtype="int32")

    def set_attrs(self):
        self.attrs = {'use_mkldnn': True}

    def set_outputs(self):
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)


class TestReshape2OneDNNOpDimInfer1_attr_OnlyShape(
        TestReshape2OneDNNOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.infered_shape = (5, -1, 10)
        self.shape = (5, -1, -1)


class TestReshapeOneDNNOp(TestReshape2OneDNNOp):
    def set_op_type(self):
        self.op_type = "reshape"

    def set_outputs(self):
        self.outputs = {"Out": self.inputs["X"].reshape(self.infered_shape)}

    def test_check_output(self):
        self.check_output()


class TestReshapeOneDNNOpDimInfer1(TestReshapeOneDNNOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOneDNNOp_attr_OnlyShape(TestReshape2OneDNNOp_attr_OnlyShape):
    def set_op_type(self):
        self.op_type = "reshape"

    def set_outputs(self):
        self.outputs = {"Out": self.inputs["X"].reshape(self.infered_shape)}

    def test_check_output(self):
        self.check_output()


class TestReshapeOneDNNOpDimInfer1_attr_OnlyShape(
        TestReshapeOneDNNOp_attr_OnlyShape):
    def init_data(self):
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.infered_shape = (5, -1, 10)
        self.shape = (5, -1, -1)


#   BF16 TESTS
def create_reshape_bf16_test_classes(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestReshape2BF16OneDNNOp(parent):
        def set_inputs(self):
            self.dtype = np.uint16
            self.inputs = {"X": convert_float_to_uint16(self.x)}

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_output(self):
            self.check_output_with_place(
                core.CPUPlace(), no_check_set=["XShape"])

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[self.dout])

    cls_name = "{0}_{1}".format(parent.__name__, "Reshape2_BF16")
    TestReshape2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestReshape2BF16OneDNNOp

    class TestReshapeBF16OneDNNOp(TestReshape2BF16OneDNNOp):
        def set_op_type(self):
            self.dtype = np.uint16
            self.op_type = "reshape"

        def set_outputs(self):
            self.outputs = {"Out": self.x.reshape(self.new_shape)}

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    cls_name = "{0}_{1}".format(parent.__name__, "Reshape_BF16")
    TestReshapeBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestReshapeBF16OneDNNOp


create_reshape_bf16_test_classes(TestReshape2OneDNNOp)
create_reshape_bf16_test_classes(TestReshape2OneDNNOpDimInfer1)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
