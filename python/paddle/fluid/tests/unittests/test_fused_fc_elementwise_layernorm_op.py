#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from paddle.fluid import core
from test_fc_op import fc_refer, MatrixGenerate
from test_layer_norm_op import _reference_layer_norm_naive

np.random.random(123)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedFCElementwiseLayerNormOp(OpTest):

    def config(self):
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3, 2)
        self.y_shape = [1, 15]
        self.begin_norm_axis = 1

    def setUp(self):
        self.op_type = "fused_fc_elementwise_layernorm"
        self.config()

        # Attr of layer_norm
        epsilon = 0.00001

        # fc
        fc_out = fc_refer(self.matrix, True, True)
        # elementwise_add
        y = np.random.random_sample(self.y_shape).astype(np.float32)
        add_out = fc_out + y
        # layer_norm
        scale_shape = [np.prod(self.y_shape[self.begin_norm_axis:])]
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias_1 = np.random.random_sample(scale_shape).astype(np.float32)
        out, mean, variance = _reference_layer_norm_naive(
            add_out, scale, bias_1, epsilon, self.begin_norm_axis)

        self.inputs = {
            "X": self.matrix.input,
            "W": self.matrix.weights,
            "Bias0": self.matrix.bias,
            "Y": y,
            "Scale": scale,
            "Bias1": bias_1
        }
        self.attrs = {
            "activation_type": "relu",
            "epsilon": epsilon,
            "begin_norm_axis": self.begin_norm_axis
        }
        self.outputs = {"Out": out, "Mean": mean, "Variance": variance}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)


class TestFusedFCElementwiseLayerNormOp2(TestFusedFCElementwiseLayerNormOp):

    def config(self):
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)
        self.y_shape = [4, 6]
        self.begin_norm_axis = 1


if __name__ == '__main__':
    unittest.main()
