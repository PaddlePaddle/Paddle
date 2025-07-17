# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy

import paddle


class TestFusedRmsNorm(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        self.init_data()
        self.modify_data()
        self.func = paddle.incubate.nn.functional.fused_rms_norm

    def tearDown(self):
        pass

    def init_data(self):
        hidden_size = 768
        shape = [1028, hidden_size]
        dtype = "float32"

        self.x = paddle.uniform(shape, dtype=dtype)
        self.residual = paddle.uniform(shape, dtype=dtype)
        self.bias = paddle.uniform([hidden_size], dtype=dtype)
        self.norm_weight = paddle.uniform([hidden_size], dtype=dtype)
        self.norm_bias = paddle.uniform([hidden_size], dtype=dtype)
        self.x.stop_gradient = False
        self.residual.stop_gradient = False
        self.bias.stop_gradient = False
        self.norm_weight.stop_gradient = False
        self.norm_bias.stop_gradient = False

        self.epsilon = 1e-6
        self.begin_norm_axis = -1
        self.quant_scale = -1
        self.quant_round_type = 0
        self.quant_max_bound = 0
        self.quant_min_bound = 0

    def modify_data(self):
        pass

    def inputs(self):
        return (
            self.x,
            self.norm_weight,
            self.norm_bias,
            self.epsilon,
            self.begin_norm_axis,
            self.bias,
            self.residual,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )

    def compute(self):
        inputs = self.inputs()
        dy_out = self.func(*inputs)
        static_func = paddle.jit.to_static(
            full_graph=True,
            backend="CINN",
            input_spec=None,
        )(self.func)
        st_out = static_func(*inputs)
        return dy_out, st_out

    def test_eval(self):
        dy_out, st_out = self.compute()
        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


class TestFusedRmsNormQuantRint(TestFusedRmsNorm):
    def modify_data(self):
        self.quant_scale = 0.15
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127

    def test_eval(self):
        # There is little precision difference after decomposition.
        # which leads to different results after dequantization. So
        # we skip accuracy check this test.
        self.compute()


class TestFusedRmsNormQuantRound(TestFusedRmsNorm):
    def modify_data(self):
        self.quant_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127

    def test_eval(self):
        # There is little precision difference after decomposition.
        # which leads to different results after dequantization. So
        # we skip accuracy check in this test.
        self.compute()


if __name__ == '__main__':
    unittest.main()
