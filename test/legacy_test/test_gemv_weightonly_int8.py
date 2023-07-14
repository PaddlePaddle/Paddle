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
import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid import core
from paddle.framework import set_default_dtype

np.random.seed(123)
paddle.seed(123)


class GemvWeightOnlyInt8TestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.in_features = 64
        self.out_features = 64
        self.algo = "weight_only"
        self.bits = 8
        self.act_method = "gelu"

    def setUp(self):
        self.config()
        input = np.random.random((1, self.in_features))
        self.input = paddle.to_tensor(input, dtype=self.dtype)
        if self.bias:
            bias_attr = fluid.ParamAttr(
                learning_rate=0.8,
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

        self.weight, self.weight_scale = F.quant_for_compress(
            self.linear.weight, bits=self.bits, layout=self.algo
        )
        self.bias = self.linear.bias
        if self.act_method != "None":
            self.activate = getattr(F, self.act_method)
        else:
            self.activate = None

    def get_baseline_out(self):
        out = self.linear(self.input)
        if self.activate:
            out = self.activate(out)
        return out

    def get_gemv_weightonly_int8(self):
        out = paddle.incubate.nn.functional.gemv_weightonly_int8(
            self.input,
            self.weight,
            self.bias,
            self.weight_scale,
            self.act_method,
        )
        return out

    def get_gemv_weightonly_int8_static(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = paddle.static.data(
                    "x", self.input.shape, dtype=self.input.dtype
                )
                weight = paddle.static.data(
                    "weight", self.weight.shape, dtype=self.weight.dtype
                )
                bias = paddle.static.data(
                    "bias", self.bias.shape, dtype=self.bias.dtype
                )
                weight_scale = paddle.static.data(
                    "weight_scale",
                    self.weight_scale.shape,
                    dtype=self.weight_scale.dtype,
                )

                out = F.grid_sample(
                    x, weight, bias, weight_scale, self.act_method
                )
        feed_dict = {
            'x': self.input,
            'weight': self.weight,
            'bias': self.bias,
            "weight_scale": self.weight_scale,
        }
        exe = fluid.Executor(paddle.GPUPlace())
        exe.run(start)
        (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        return out

    def test_gemv_weightonly_int8(self):
        out_real = self.get_gemv_weightonly_int8()
        out_expect = self.get_baseline_out()
        out_real = convert_uint16_to_float(out_real)
        out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


class GemvWeightOnlyInt8TestCaseCase1(GemvWeightOnlyInt8TestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = True
        self.in_features = 128
        self.out_features = 64


class GemvWeightOnlyInt8TestCaseCase2(GemvWeightOnlyInt8TestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.in_features = 64
        self.out_features = 64


@unittest.skipIf(
    not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not support bfloat16",
)
class GemvWeightOnlyInt8TestCaseCase3(GemvWeightOnlyInt8TestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.bias = False
        self.in_features = 64
        self.out_features = 64
        self.atol = 1e-1


class GemvWeightOnlyInt8TestCaseCase4(GemvWeightOnlyInt8TestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = True
        self.in_features = 128
        self.out_features = 64
        self.act_method = "None"


class GemvWeightOnlyInt8TestStatic(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.in_features = 64
        self.out_features = 64
        self.algo = "weight_only"
        self.bits = 8
        self.act_method = "gelu"

    def setUp(self):
        self.config()
        input = np.random.random((1, self.in_features))
        self.input = paddle.to_tensor(input, dtype=self.dtype)
        if self.bias:
            bias_attr = fluid.ParamAttr(
                learning_rate=0.8,
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

        self.weight, self.weight_scale = F.quant_for_compress(
            self.linear.weight, bits=self.bits, layout=self.algo
        )
        self.bias = self.linear.bias
        if self.act_method != "None":
            self.activate = getattr(F, self.act_method)
        else:
            self.activate = None

    def get_baseline_out(self):
        out = self.linear(self.input)
        if self.activate:
            out = self.activate(out)
        return out

    def get_gemv_weightonly_int8_static(self):
        paddle.enable_static()
        main = fluid.Program()
        start = fluid.Program()
        with fluid.program_guard(main, start):
            x = paddle.static.data(
                "x", self.input.shape, dtype=self.input.dtype
            )
            weight = paddle.static.data(
                "weight", self.weight.shape, dtype=self.weight.dtype
            )
            bias = paddle.static.data(
                "bias", self.bias.shape, dtype=self.bias.dtype
            )
            weight_scale = paddle.static.data(
                "weight_scale",
                self.weight_scale.shape,
                dtype=self.weight_scale.dtype,
            )

            out = paddle.incubate.nn.functional.gemv_weightonly_int8(
                x, weight, bias, weight_scale, self.act_method
            )
            feed_dict = {
                'x': self.input.numpy(),
                'weight': self.weight.numpy(),
                'bias': self.bias.numpy(),
                "weight_scale": self.weight_scale.numpy(),
            }
            exe = fluid.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        return out

    def test_gemv_weightonly_int8(self):
        out_expect = self.get_baseline_out()
        out_real = self.get_gemv_weightonly_int8_static()
        out_real = convert_uint16_to_float(out_real)
        out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()
