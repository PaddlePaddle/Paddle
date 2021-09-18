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
import copy
import collections
import numpy as np

from paddle.framework import ParamAttr
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.fluid.core as core
from paddle.nn.layer import transformer
import paddle.nn.functional as F
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout

import unittest


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedFFNOp(unittest.TestCase):
    def getDtype(self):
        self.dtype = "float32"
        self.layer_norm_dtype = "float32"

    def getShape(self):
        self.batch_size = np.random.randint(1, 64)
        self.query_length = np.random.randint(32, 256)
        self.d_model = np.random.randint(32, 1024)
        self.dim_feedforward = np.random.randint(32, 4096)

    def getDiff(self):
        self.rtol = 1e-4
        self.atol = 1e-5

    def getActivation(self):
        self.act_method = "gelu"

    def getNormalizeBefore(self):
        self.normalize_before = False

    def setUp(self):
        paddle.disable_static()
        self.getDtype()
        self.getShape()
        self.getDiff()
        self.getActivation()
        self.getNormalizeBefore()
        paddle.set_default_dtype(self.dtype)
        self.weight_attr = None
        self.bias_attr = None

        self.weight_attrs = transformer._convert_param_attr_to_list(
            self.weight_attr, 2)
        self.bias_attrs = transformer._convert_param_attr_to_list(
            self.bias_attr, 2)
        self.linear1 = Linear(
            self.d_model,
            self.dim_feedforward,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1])
        self.linear2 = Linear(
            self.dim_feedforward,
            self.d_model,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1])

        paddle.set_default_dtype(self.layer_norm_dtype)
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.dropout = Dropout(0.0, mode="upscale_in_train")
        self.dropout1 = Dropout(0.0, mode="upscale_in_train")
        self.dropout2 = Dropout(0.0, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

        self.src = np.random.random((self.batch_size, self.query_length,
                                     self.d_model)).astype(self.dtype)

    def Base(self):
        paddle.disable_static()
        tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
        residual = paddle.to_tensor(self.src)
        if self.normalize_before:
            ln1_out = self.norm1(tensor_src)
            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(ln1_out))))
            dropout2_out = residual + self.dropout2(linear2_out)
        else:
            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(tensor_src))))
            dropout2_out = residual + self.dropout2(linear2_out)
            dropout2_out = self.norm2(dropout2_out)
        return dropout2_out

    def FusedFFN(self):
        paddle.disable_static()
        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            linear1_weight = paddle.to_tensor(
                self.linear1.weight, stop_gradient=False)
            linear1_bias = paddle.to_tensor(
                self.linear1.bias, stop_gradient=False)
            linear2_weight = paddle.to_tensor(
                self.linear2.weight, stop_gradient=False)
            linear2_bias = paddle.to_tensor(
                self.linear2.bias, stop_gradient=False)
            ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
            ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
            ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
            ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)
            x = paddle.to_tensor(self.src, stop_gradient=False)
            out = F.fused_ffn(
                x,
                linear1_weight,
                linear2_weight,
                linear1_bias,
                linear2_bias,
                ln1_scale,
                ln1_bias,
                ln2_scale,
                ln2_bias,
                0.0,
                0.0,
                act_method=self.act_method,
                normalize_pre_or_post=self.normalize_before)
            return out

    def test_fused_ffn(self):
        base_out = self.Base()
        fused_out = self.FusedFFN()
        np.testing.assert_allclose(
            base_out.numpy(), fused_out.numpy(), rtol=self.rtol, atol=self.atol)


class TestFusedFFNOpFp16(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float16"
        self.layer_norm_dtype = "float32"

    def getDiff(self):
        self.rtol = 1e-2
        self.atol = 1e-3

    def getShape(self):
        self.batch_size = 1
        self.query_length = 8
        self.d_model = 8
        self.dim_feedforward = 8


class TestFusedFFNOpFp64(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float64"
        self.layer_norm_dtype = "float64"


class TestFusedFFNOpActivation(TestFusedFFNOp):
    def getActivation(self):
        self.act_method = "relu"


class TestFusedFFNOpNormalizeBefore(TestFusedFFNOp):
    def getNormalizeBefore(self):
        self.normalize_before = True

    def getShape(self):
        self.batch_size = 1
        self.query_length = 1
        self.d_model = 8
        self.dim_feedforward = 8


if __name__ == "__main__":
    unittest.main()
