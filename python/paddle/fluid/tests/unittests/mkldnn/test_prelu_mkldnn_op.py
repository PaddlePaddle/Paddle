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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16


def ref_prelu(x, weight, mode):
    result = x.copy()

    if mode == "all":
        result = np.where(x > 0, x, x * weight[0])
    elif mode == "channel":
        if len(weight.shape) > 1:
            for i in range(x.shape[1]):
                result[:, i] = np.where(x[:, i] > 0, x[:, i],
                                        x[:, i] * weight[0, i])
        else:
            for i in range(x.shape[1]):
                result[:, i] = np.where(x[:, i] > 0, x[:, i],
                                        x[:, i] * weight[i])
    elif mode == "element":
        result = np.where(x[:] > 0, x[:], x[:] * weight)

    return result


class TestPReluModeChannelOneDNNOp(OpTest):

    def init_attrs(self):
        self.mode = "element"
        self.alpha = np.random.random((1, 4, 5, 5)).astype("float32")

    def set_dtype_attr(self):
        pass

    def set_inputs(self):
        self.inputs = {'X': self.x, 'Alpha': self.alpha}

    def setUp(self):
        self.op_type = "prelu"
        self.x = np.random.random((2, 4, 5, 5)).astype("float32") + 1
        self.init_attrs()
        self.set_inputs()
        self.attrs = {'mode': self.mode, 'use_mkldnn': True}
        self.set_dtype_attr()

        self.outputs = {'Out': ref_prelu(self.x, self.alpha, self.mode)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Alpha'], 'Out')


class TestPReluModeAllOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        self.mode = "all"
        self.alpha = np.random.random((1, 1, 1, 1)).astype("float32")

    # Skip 'Alpha' input check because in mode = 'all' it has to be a single
    # 1D value so checking if it has at least 100 values will cause an error
    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPReluModeElementOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        self.mode = "element"
        self.alpha = np.random.random((1, 4, 5, 5)).astype("float32")


class TestPReluModeChannel3DOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        self.mode = "channel"
        self.x = np.random.random((1, 100, 1)).astype("float32")
        self.alpha = np.random.random((1, 100, 1)).astype("float32")


class TestPReluModeChannelAlpha1DOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        self.mode = "channel"
        self.x = np.random.random((1, 100, 1)).astype("float32")
        self.alpha = np.random.random((100)).astype("float32")


class TestPReluModeAllAlpha1DOneDNNOp(TestPReluModeAllOneDNNOp):

    def init_attrs(self):
        self.mode = "channel"
        self.x = np.random.random((1, 1, 100)).astype("float32")
        self.alpha = np.random.random((1)).astype("float32")


#   BF16 TESTS
def create_bf16_test_class(parent):

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestPReluBF16OneDNNOp(parent):

        def set_inputs(self, ):
            self.inputs = {
                'X': convert_float_to_uint16(self.x),
                'Alpha': convert_float_to_uint16(self.alpha)
            }

        def set_dtype_attr(self):
            self.attrs['mkldnn_data_type'] = "bfloat16"

        def calculate_grads(self):
            dout = self.outputs['Out']
            self.dx = self.x.copy()
            self.dalpha = self.alpha.copy()

            if self.mode == "all":
                self.dx = np.where(self.x > 0, dout, dout * self.alpha[0])
            elif self.mode == "channel":
                if len(self.alpha.shape) > 1:
                    for i in range(self.x.shape[1]):
                        self.dx[:, i] = np.where(self.x[:, i] > 0, dout[:, i],
                                                 dout[:, i] * self.alpha[0, i])
                else:
                    for i in range(self.x.shape[1]):
                        self.dx[:, i] = np.where(self.x[:, i] > 0, dout[:, i],
                                                 dout[:, i] * self.alpha[i])
                    self.dx
            elif self.mode == "element":
                self.dx = np.where(self.x[:] > 0, dout[:], dout[:] * self.alpha)

            self.dalpha = np.where(self.x < 0, dout * self.x, 0)
            self.dout = dout

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X", "Alpha"],
                "Out",
                user_defined_grads=[self.dx, self.dalpha],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    cls_name = "{0}_{1}".format(parent.__name__, "BF16")
    TestPReluBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestPReluBF16OneDNNOp


create_bf16_test_class(TestPReluModeChannelOneDNNOp)
create_bf16_test_class(TestPReluModeElementOneDNNOp)
create_bf16_test_class(TestPReluModeChannel3DOneDNNOp)
create_bf16_test_class(TestPReluModeChannelAlpha1DOneDNNOp)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
