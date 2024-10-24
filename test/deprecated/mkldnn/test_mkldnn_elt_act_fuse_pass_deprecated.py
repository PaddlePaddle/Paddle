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

import sys
import unittest

import numpy as np

sys.path.append("../../ir/inference")
from inference_pass_test import InferencePassTest

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base.core import PassVersionChecker


class ElementwiseActivationOneDNNFusePassTest(InferencePassTest):
    act_alpha = None
    act_beta = None
    pass_name = 'elementwise_act_onednn_fuse_pass'

    def setUp(self):
        self.set_params()
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(self.main_program, self.startup_program):
                data_A = paddle.static.data(
                    name="data_A", shape=[-1, 3, 100, 100], dtype="float32"
                )
                data_B = paddle.static.data(
                    name="data_B", shape=[-1, 3, 100, 100], dtype="float32"
                )
                elt_out = self.operand(data_A, data_B)
                if self.act is not None:
                    if self.act_beta is not None:
                        elt_out = self.act(
                            elt_out, self.act_alpha, self.act_beta
                        )
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
        with paddle.pir_utils.OldIrGuard():
            self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ElementwiseActivationOneDNNFusePassTest_Add_Relu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = F.relu


class ElementwiseActivationOneDNNFusePassTest_Add_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.tanh


class ElementwiseActivationOneDNNFusePassTest_Add_LeakyRelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationOneDNNFusePassTest_Add_Swish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.swish


class ElementwiseActivationOneDNNFusePassTest_Add_HardSwish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationOneDNNFusePassTest_Add_SQRT(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.sqrt


class ElementwiseActivationOneDNNFusePassTest_Add_ABS(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.abs


class ElementwiseActivationOneDNNFusePassTest_Add_Clip(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationOneDNNFusePassTest_Add_Gelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationOneDNNFusePassTest_Add_Gelu_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationOneDNNFusePassTest_Add_Relu6(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationOneDNNFusePassTest_Add_Sigmoid(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationOneDNNFusePassTest_Sub_Relu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = F.relu


class ElementwiseActivationOneDNNFusePassTest_Sub_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.tanh


class ElementwiseActivationOneDNNFusePassTest_Sub_LeakyRelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationOneDNNFusePassTest_Sub_Swish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.swish


class ElementwiseActivationOneDNNFusePassTest_Sub_HardSwish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationOneDNNFusePassTest_Sub_ABS(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.abs


class ElementwiseActivationOneDNNFusePassTest_Sub_Clip(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationOneDNNFusePassTest_Sub_Gelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationOneDNNFusePassTest_Sub_Gelu_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationOneDNNFusePassTest_Sub_Relu6(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationOneDNNFusePassTest_Sub_Sigmoid(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationOneDNNFusePassTest_Mul_Relu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = F.relu


class ElementwiseActivationOneDNNFusePassTest_Mul_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.tanh


class ElementwiseActivationOneDNNFusePassTest_Mul_LeakyRelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationOneDNNFusePassTest_Mul_Swish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.swish


class ElementwiseActivationOneDNNFusePassTest_Mul_HardSwish(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.hardswish


class ElementwiseActivationOneDNNFusePassTest_Mul_SQRT(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.sqrt


class ElementwiseActivationOneDNNFusePassTest_Mul_ABS(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.abs


class ElementwiseActivationOneDNNFusePassTest_Mul_Clip(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationOneDNNFusePassTest_Mul_Gelu(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationOneDNNFusePassTest_Mul_Gelu_Tanh(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationOneDNNFusePassTest_Mul_Relu6(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationOneDNNFusePassTest_Mul_Sigmoid(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act = paddle.nn.functional.sigmoid


class ElementwiseScaleOneDNNFusePassTest_Add(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.add
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Sub(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.subtract
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Mul(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.multiply
        self.act_alpha = 0.6
        self.act = paddle.scale


class ElementwiseScaleOneDNNFusePassTest_Div(
    ElementwiseActivationOneDNNFusePassTest
):
    def set_params(self):
        self.operand = paddle.divide
        self.act_alpha = 0.6
        self.act = paddle.scale


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
