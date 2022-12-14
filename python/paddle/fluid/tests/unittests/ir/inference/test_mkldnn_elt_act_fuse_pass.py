# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.core import PassVersionChecker


class ElementwiseActivationMkldnnFusePassTest(InferencePassTest):
    act_alpha = None
    act_beta = None
    pass_name = 'elt_act_mkldnn_fuse_pass'

    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data_A = fluid.data(
                name="data_A", shape=[-1, 3, 100, 100], dtype="float32"
            )
            data_B = fluid.data(
                name="data_B", shape=[-1, 3, 100, 100], dtype="float32"
            )
            elt_out = self.operand(data_A, data_B)
            if self.act is not None:
                if self.act_beta is not None:
                    elt_out = self.act(elt_out, self.act_alpha, self.act_beta)
                elif self.act_alpha is not None:
                    elt_out = self.act(elt_out, self.act_alpha)
                else:
                    elt_out = self.act(elt_out)

        self.feeds = {
            "data_A": np.random.random((1, 3, 100, 100)).astype("float32"),
            "data_B": np.random.random((1, 3, 100, 100)).astype("float32"),
        }
        self.fetch_list = [elt_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.operand = paddle.add
        self.act = None

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ElementwiseActivationMkldnnFusePassTest_Add_Relu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = F.relu


class ElementwiseActivationMkldnnFusePassTest_Add_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Add_LeakyRelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Add_Swish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Add_HardSwish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationMkldnnFusePassTest_Add_SQRT(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.sqrt


class ElementwiseActivationMkldnnFusePassTest_Add_ABS(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Add_Clip(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Add_Gelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Add_Gelu_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Add_Relu6(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Add_Sigmoid(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationMkldnnFusePassTest_Sub_Relu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = F.relu


class ElementwiseActivationMkldnnFusePassTest_Sub_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Sub_LeakyRelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Sub_Swish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Sub_HardSwish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationMkldnnFusePassTest_Sub_ABS(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Sub_Clip(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Sub_Gelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Sub_Gelu_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Sub_Relu6(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Sub_Sigmoid(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationMkldnnFusePassTest_Mul_Relu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = F.relu


class ElementwiseActivationMkldnnFusePassTest_Mul_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Mul_LeakyRelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Mul_Swish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Mul_HardSwish(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationMkldnnFusePassTest_Mul_SQRT(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.sqrt


class ElementwiseActivationMkldnnFusePassTest_Mul_ABS(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Mul_Clip(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Mul_Gelu(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Mul_Gelu_Tanh(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Mul_Relu6(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Mul_Sigmoid(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.sigmoid


class ElementwiseScaleOneDNNFusePassTest_Add(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Sub(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Mul(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Div(
    ElementwiseActivationMkldnnFusePassTest
):
    def set_params(self):
        self.operand = paddle.divide
        self.act_alpha = 0.6
        self.act = paddle.scale


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
