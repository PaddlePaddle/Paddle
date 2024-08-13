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
from op_test import OpTest, OpTestTool, convert_float_to_uint16

import paddle
from paddle.base import core


@OpTestTool.skip_if_not_cpu_bf16()
class TestFlattenOneDNNOp(OpTest):
    def setUp(self):
        self.set_op_type()
        self.init_test_case()
        self.set_inputs()
        self.attrs = {"axis": self.axis, 'use_mkldnn': True}
        self.ori_shape = self.inputs['X'].shape
        self.outputs = {"Out": self.inputs["X"].copy().reshape(self.new_shape)}

    def set_inputs(self):
        self.inputs = {"X": np.random.random(self.in_shape).astype("float32")}

    def set_op_type(self):
        self.op_type = "flatten"

    def test_check_output(self):
        self.check_output_with_place(
            core.CPUPlace(), check_pir_onednn=(self.op_type == "flatten2")
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CPUPlace(),
            ["X"],
            "Out",
            check_pir_onednn=(self.op_type == "flatten2"),
        )

    def init_test_case(self):
        self.in_shape = (3, 2, 2, 10)
        self.axis = 1
        self.new_shape = (3, 40)


class TestFlattenOneDNNOp1(TestFlattenOneDNNOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 2, 10)
        self.axis = 0
        self.new_shape = (1, 120)


class TestFlattenOneDNNOpSixDims(TestFlattenOneDNNOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.axis = 4
        self.new_shape = (36, 16)


class TestFlatten2OneDNNOp(TestFlattenOneDNNOp):
    def set_op_type(self):
        self.op_type = "flatten2"


class TestFlatten2OneDNNOp1(TestFlattenOneDNNOp1):
    def set_op_type(self):
        self.op_type = "flatten2"


class TestFlatten2OneDNNOpSixDims(TestFlattenOneDNNOpSixDims):
    def set_op_type(self):
        self.op_type = "flatten2"


#   BF16 TESTS
def create_flatten_bf16_test_classes(parent):
    class TestFlatten2BF16OneDNNOp(parent):
        def set_inputs(self):
            self.dtype = np.uint16
            self.inputs = {
                "X": np.random.random(self.in_shape).astype("uint16")
            }

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_output(self):
            self.check_output_with_place(
                core.CPUPlace(),
                no_check_set=["XShape"],
                check_pir_onednn=(self.op_type == "flatten2"),
            )

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[self.dout],
                check_pir_onednn=(self.op_type == "flatten2"),
            )

    cls_name = "{}_{}".format(parent.__name__, "Flatten2_BF16")
    TestFlatten2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestFlatten2BF16OneDNNOp

    class TestFlattenBF16OneDNNOp(parent):
        def set_op_type(self):
            self.dtype = np.uint16
            self.op_type = "flatten"

        def set_inputs(self):
            self.dtype = np.uint16
            self.inputs = {
                "X": np.random.random(self.in_shape).astype("uint16")
            }

        def set_outputs(self):
            self.outputs = {"Out": self.x.reshape(self.new_shape)}

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_output(self):
            self.check_output_with_place(
                core.CPUPlace(), check_pir_onednn=(self.op_type == "flatten2")
            )

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)],
                check_pir_onednn=(self.op_type == "flatten2"),
            )

    cls_name = "{}_{}".format(parent.__name__, "Flatten_BF16")
    TestFlattenBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestFlattenBF16OneDNNOp


create_flatten_bf16_test_classes(TestFlatten2OneDNNOp)
create_flatten_bf16_test_classes(TestFlatten2OneDNNOp1)
create_flatten_bf16_test_classes(TestFlatten2OneDNNOpSixDims)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
