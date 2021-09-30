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

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.nn.layer import transformer
import paddle.nn.functional as F
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
import unittest
from op_test import OpTest


def Base(x_data, linear1_weight_data, linear1_bias_data, linear2_weight_data,
         linear2_bias_data, ln1_weight_data, ln1_bias_data, ln2_weight_data,
         ln2_bias_data, d_model):
    paddle.disable_static()
    x = paddle.to_tensor(x_data)
    linear1_weight = paddle.to_tensor(linear1_weight_data)
    linear1_bias = paddle.to_tensor(linear1_bias_data)
    linear2_weight = paddle.to_tensor(linear2_weight_data)
    linear2_bias = paddle.to_tensor(linear2_bias_data)
    ln1_weight = paddle.to_tensor(ln1_weight_data)
    ln1_bias = paddle.to_tensor(ln1_bias_data)
    ln2_weight = paddle.to_tensor(ln2_weight_data)
    ln2_bias = paddle.to_tensor(ln2_bias_data)
    ######base ffn######
    linear1_out = F.linear(x, linear1_weight, linear1_bias)
    act_out = F.relu(linear1_out)
    dropout1_out = F.dropout(x=act_out, p=0.0, training=False)
    linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
    dropout2_out = x + F.dropout(x=linear2_out, p=0.0, training=False)
    ln_out = F.layer_norm(
        dropout2_out,
        normalized_shape=list([d_model]),
        weight=ln2_weight,
        bias=ln2_bias)
    ######base ffn######
    return ln_out


class TestFusedFFNOp(OpTest):
    def getDtype(self):
        self.dtype = "float32"
        self.layer_norm_dtype = "float32"

    def getShape(self):
        self.batch_size = np.random.randint(1, 64)
        self.query_length = np.random.randint(32, 256)
        self.d_model = np.random.randint(32, 1024)
        self.dim_feedforward = np.random.randint(32, 4096)

    def getDiff(self):
        self.rtol = 1e-3
        self.atol = 1e-4

    def getActivation(self):
        self.act_method = "gelu"

    def getNormalizeBefore(self):
        self.normalize_before = False

    def setUp(self):
        paddle.disable_static()
        self.__class__.op_type = "fused_feedforward"
        self.getDtype()
        self.getShape()
        self.getDiff()
        self.getActivation()
        self.getNormalizeBefore()
        paddle.set_default_dtype(self.dtype)
        self.weight_attr = None
        self.bias_attr = None
        self.x = np.random.random((self.batch_size, self.query_length,
                                   self.d_model)).astype(self.dtype)
        self.linear1_weight = np.random.random(
            (self.d_model, self.dim_feedforward)).astype(self.dtype)
        self.linear1_bias = np.random.random(
            (self.dim_feedforward)).astype(self.dtype)
        self.linear2_weight = np.random.random(
            (self.dim_feedforward, self.d_model)).astype(self.dtype)
        self.linear2_bias = np.random.random((self.d_model)).astype(self.dtype)
        self.ln1_weight = np.random.random(
            (self.d_model)).astype(self.layer_norm_dtype)
        self.ln1_bias = np.random.random(
            (self.d_model)).astype(self.layer_norm_dtype)
        self.ln2_weight = np.random.random(
            (self.d_model)).astype(self.layer_norm_dtype)
        self.ln2_bias = np.random.random(
            (self.d_model)).astype(self.layer_norm_dtype)

        self.inputs = {
            'X': self.x,
            'Linear1Weight': self.linear1_weight,
            'Linear1Bias': self.linear1_bias,
            'Linear2Weight': self.linear2_weight,
            'Linear2Bias': self.linear2_bias,
            'Ln1Scale': self.ln1_weight,
            'Ln1Bias': self.ln1_bias,
            'Ln2Scale': self.ln2_weight,
            'Ln2Bias': self.ln2_bias
        }
        self.attrs = {
            'dropout1_prob': 0.0,
            'dropout2_prob': 0.0,
            'act_method': self.act_method,
            'normalized_pre_or_post': self.normalize_before
        }
        self.base_out = Base(self.x, self.linear1_weight, self.linear1_bias,
                             self.linear2_weight, self.linear2_bias,
                             self.ln1_weight, self.ln1_bias, self.ln2_weight,
                             self.ln2_bias, self.d_model)
        self.outputs = {'Out': self.base_out}

    def _get_places(self):
        places = [fluid.CUDAPlace(0)]
        return places

    def test_output(self):
        paddle.enable_static()
        self.check_output()

    #def test_grad(self):
    #    self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
