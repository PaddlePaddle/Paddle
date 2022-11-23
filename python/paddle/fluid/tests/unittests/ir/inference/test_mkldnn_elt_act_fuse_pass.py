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
from paddle.fluid.core import PassVersionChecker


class ElementwiseActivationMkldnnFusePassTest(InferencePassTest):
    act_alpha = None
    act_beta = None
    pass_name = 'elt_act_mkldnn_fuse_pass'

    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
<<<<<<< HEAD
            data_A = fluid.data(name="data_A",
                                shape=[-1, 3, 100, 100],
                                dtype="float32")
            data_B = fluid.data(name="data_B",
                                shape=[-1, 3, 100, 100],
                                dtype="float32")
=======
            data_A = fluid.data(
                name="data_A", shape=[-1, 3, 100, 100], dtype="float32"
            )
            data_B = fluid.data(
                name="data_B", shape=[-1, 3, 100, 100], dtype="float32"
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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
        self.operand = fluid.layers.elementwise_add
        self.act = None

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ElementwiseActivationMkldnnFusePassTest_Add_Relu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = fluid.layers.relu


class ElementwiseActivationMkldnnFusePassTest_Add_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Add_LeakyRelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Add_Swish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Add_HardSwish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = fluid.layers.hard_swish


class ElementwiseActivationMkldnnFusePassTest_Add_SQRT(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.sqrt


class ElementwiseActivationMkldnnFusePassTest_Add_ABS(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Add_Clip(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Add_Gelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Add_Gelu_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Add_Relu6(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Add_Sigmoid(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_add
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationMkldnnFusePassTest_Sub_Relu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = fluid.layers.relu


class ElementwiseActivationMkldnnFusePassTest_Sub_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Sub_LeakyRelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Sub_Swish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Sub_HardSwish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = fluid.layers.hard_swish


class ElementwiseActivationMkldnnFusePassTest_Sub_ABS(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Sub_Clip(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Sub_Gelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Sub_Gelu_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Sub_Relu6(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Sub_Sigmoid(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_sub
        self.act = paddle.nn.functional.sigmoid


class ElementwiseActivationMkldnnFusePassTest_Mul_Relu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = fluid.layers.relu


class ElementwiseActivationMkldnnFusePassTest_Mul_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.tanh


class ElementwiseActivationMkldnnFusePassTest_Mul_LeakyRelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act_alpha = 0.2
        self.act = paddle.nn.functional.leaky_relu


class ElementwiseActivationMkldnnFusePassTest_Mul_Swish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.nn.functional.swish


class ElementwiseActivationMkldnnFusePassTest_Mul_HardSwish(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = fluid.layers.hard_swish


class ElementwiseActivationMkldnnFusePassTest_Mul_SQRT(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.sqrt


class ElementwiseActivationMkldnnFusePassTest_Mul_ABS(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.abs


class ElementwiseActivationMkldnnFusePassTest_Mul_Clip(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = fluid.layers.clip
        self.act_alpha = 0.0
        self.act_beta = 10.0


class ElementwiseActivationMkldnnFusePassTest_Mul_Gelu(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.nn.functional.gelu


class ElementwiseActivationMkldnnFusePassTest_Mul_Gelu_Tanh(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.nn.functional.gelu
        self.act_alpha = True


class ElementwiseActivationMkldnnFusePassTest_Mul_Relu6(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.nn.functional.relu6


class ElementwiseActivationMkldnnFusePassTest_Mul_Sigmoid(
<<<<<<< HEAD
        ElementwiseActivationMkldnnFusePassTest):

=======
    ElementwiseActivationMkldnnFusePassTest
):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_params(self):
        self.operand = fluid.layers.elementwise_mul
        self.act = paddle.nn.functional.sigmoid


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
