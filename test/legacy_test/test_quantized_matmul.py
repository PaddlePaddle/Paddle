# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import convert_uint16_to_float

import paddle
import paddle.incubate.nn.functional as F
from paddle import fluid
from paddle.fluid.framework import default_main_program
from paddle.framework import set_default_dtype

np.random.seed(123)
paddle.seed(123)
default_main_program().random_seed = 42
quant_method_list = [
    "weight_only_int8",
    "weight_only_int4",
    "llm.int8",
    "None",
]


class QuantizedMatmulTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.quant_method = "None"

    def setUp(self):
        self.config()
        if self.dtype == "bfloat16":
            self.atol = 1e-1
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        if self.bias:
            bias_attr = fluid.ParamAttr(
                trainable=False,
                regularizer=None,
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
        else:
            bias_attr = None
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(
            self.in_features, self.out_features, bias_attr=bias_attr
        )

        self.bias = self.linear.bias
        self.weight = self.linear.weight
        self.weight_scale = None
        if self.quant_method in quant_method_list[0:3]:
            self.weight, self.weight_scale = F.quant_for_compress(
                self.weight, layout=self.quant_method
            )

    def get_linear_out(self):
        out = self.linear(self.x)
        return out.numpy()

    def get_quantized_matmul_out(self):
        # print("x:", self.x)
        # print("weight:", self.weight)
        # print("weight_scale:", self.weight_scale)
        # print("bias:", self.bias)
        out = F.quantized_matmul(
            self.x,
            self.weight,
            bias=self.bias,
            weight_scale=self.weight_scale,
            quant_method=self.quant_method,
        )
        return out.numpy()

    def test_quantized_matmul(self):
        out_real = self.get_quantized_matmul_out()
        out_expect = self.get_linear_out()
        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


class QuantizedMatmulTestCase1(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'


class QuantizedMatmulTestCase2(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False


class QuantizedMatmulTestCase3(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'


class QuantizedMatmulTestCase4(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"


class QuantizedMatmulTestCase5(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.bias = False


class QuantizedMatmulTestCase6(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int8"


class QuantizedMatmulTestCase7(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int4"
        self.atol = 1e-1


class QuantizedMatmulTestCase8(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int4"
        self.bias = False
        self.atol = 1e-1


class QuantizedMatmulTestCase9(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int4"


class QuantizedMatmulTestCase10(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1


class QuantizedMatmulTestCase11(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1
        self.bias = False


class QuantizedMatmulTestCase12(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1


class QuantizedMatmulTestCase13(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "llm.int8"


class QuantizedMatmulTestCase14(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "llm.int8"
        self.bias = False


class QuantizedMatmulTestCase15(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "llm.int8"


if __name__ == '__main__':
    unittest.main()
