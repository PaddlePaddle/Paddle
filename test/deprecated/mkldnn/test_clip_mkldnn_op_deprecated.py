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


class TestClipOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "clip"
        self.init_shape()
        self.set_inputs()
        self.set_attrs()
        self.set_additional_inputs()
        self.adjust_op_settings()

        self.min = (
            self.attrs['min']
            if 'Min' not in self.inputs
            else self.inputs['Min']
        )
        self.max = (
            self.attrs['max']
            if 'Max' not in self.inputs
            else self.inputs['Max']
        )

        self.outputs = {'Out': np.clip(self.x_fp32, self.min, self.max)}

    def init_shape(self):
        self.shape = [10, 10]

    def set_inputs(self):
        self.inputs = {
            'X': np.array(np.random.random(self.shape).astype(np.float32) * 25)
        }
        self.x_fp32 = self.inputs['X']

    def set_additional_inputs(self):
        pass

    def adjust_op_settings(self):
        pass

    def set_attrs(self):
        self.attrs = {'min': 7.2, 'max': 9.6, 'use_mkldnn': True}

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestClipOneDNNOp_ZeroDim(TestClipOneDNNOp):
    def init_shape(self):
        self.shape = []


class TestClipMinAsInputOneDNNOp(TestClipOneDNNOp):
    def set_additional_inputs(self):
        self.inputs['Min'] = np.array([6.8]).astype('float32')


class TestClipMaxAsInputOneDNNOp(TestClipOneDNNOp):
    def set_additional_inputs(self):
        self.inputs['Max'] = np.array([9.1]).astype('float32')


class TestClipMaxAndMinAsInputsOneDNNOp(TestClipOneDNNOp):
    def set_additional_inputs(self):
        self.inputs['Max'] = np.array([8.5]).astype('float32')
        self.inputs['Min'] = np.array([7.1]).astype('float32')


#   BF16 TESTS
def create_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestClipBF16OneDNNOp(parent):
        def set_inputs(self):
            self.x_fp32 = np.random.random((10, 10)).astype(np.float32) * 25
            self.inputs = {'X': convert_float_to_uint16(self.x_fp32)}

        def adjust_op_settings(self):
            self.dtype = np.uint16
            self.attrs['mkldnn_data_type'] = "bfloat16"

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = np.zeros(self.x_fp32.shape).astype("float32")

            for i in range(self.dx.shape[0]):
                for j in range(self.dx.shape[1]):
                    if (
                        self.x_fp32[j][i] > self.min
                        and self.x_fp32[j][i] < self.max
                    ):
                        self.dx[j][i] = self.dout[j][i]

        def test_check_output(self):
            self.check_output_with_place(
                core.CPUPlace(), check_dygraph=False, check_pir_onednn=True
            )

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)],
                check_dygraph=False,
                check_pir_onednn=True,
            )

    cls_name = "{}_{}".format(parent.__name__, "BF16")
    TestClipBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestClipBF16OneDNNOp


create_bf16_test_class(TestClipOneDNNOp)
create_bf16_test_class(TestClipMinAsInputOneDNNOp)
create_bf16_test_class(TestClipMaxAsInputOneDNNOp)
create_bf16_test_class(TestClipMaxAndMinAsInputsOneDNNOp)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
