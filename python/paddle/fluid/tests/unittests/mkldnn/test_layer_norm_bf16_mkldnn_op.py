#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# from paddle.fluid.tests.unittests.test_layer_norm_op import *
from __future__ import print_function
import unittest
import numpy as np

from operator import mul
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle import enable_static
from functools import reduce

from paddle.fluid.tests.unittests.mkldnn.test_layer_norm_mkldnn_op import _reference_layer_norm_naive
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool

from paddle_bfloat import bfloat16


@OpTestTool.skip_if_not_cpu_bf16()
class TestLayerNormBF16MKLDNNOp(OpTest):

    def setUp(self):
        self.op_type = "layer_norm"
        self.config()

        x = np.random.random(self.input_shape).astype(np.float32)
        self.inputs = {'X': x.astype("bfloat16")}

        if self.with_scale_and_bias:
            scale = np.random.random_sample(self.scale_shape).astype(np.float32)
            bias = np.random.random_sample(self.scale_shape).astype(np.float32)

            self.inputs['Scale'] = scale
            self.inputs['Bias'] = bias
        else:
            scale = np.array([])
            bias = np.array([])

        self.attrs = {
            "epsilon": self.epsilon,
            "begin_norm_axis": self.begin_norm_axis,
            "use_mkldnn": True,
            "is_test": self.is_test
        }

        output, mean, var = _reference_layer_norm_naive(x, scale, bias,
                                                        self.epsilon,
                                                        self.begin_norm_axis)
        self.outputs = {
            'Y': output.astype('bfloat16'),
        }

        if self.is_test == False:
            self.outputs['Mean'] = mean
            self.outputs['Variance'] = var

    def config(self):
        self.input_shape = (2, 3, 4, 5)
        self.scale_shape = [5]
        self.epsilon = 1e-4
        self.is_test = True
        self.with_scale_and_bias = False
        self.begin_norm_axis = 3

    # TODO (jczaja): Enable testing without is_test and with scale
    # and bias when enabling layer_norm for training using bf16
    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


if __name__ == "__main__":
    enable_static()
    unittest.main()
