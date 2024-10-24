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
from test_weight_only_linear import convert_uint16_to_float, get_cuda_version

import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.framework import set_default_dtype


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-1
        self.bias = True
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 128
        self.threshold = 6.0
        self.static = False

    def setUp(self):
        np.random.seed(123)
        paddle.seed(42)
        self.config()
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        if self.bias:
            bias_attr = base.ParamAttr(
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

        self.weight = self.linear.weight
        self.weight_scale = None
        self.weight, self.weight_scale = Q.weight_quantize(
            self.weight, algo="llm.int8"
        )

    def dynamic_quant(self, x):
        row_ranges = paddle.max(x, axis=[-1]).astype('float32')
        row_ranges = row_ranges.unsqueeze(-1)
        quant_x = paddle.round(
            paddle.clip(
                x.astype('float32') * 127.0 * (1 / row_ranges),
                min=-127.0,
                max=127.0,
            )
        ).astype('int8')
        return quant_x, row_ranges

    def get_linear_out(self):
        outlier_cols = (
            paddle.nonzero(paddle.max(self.x, axis=[0, 1]) > self.threshold)
            .reshape([-1])
            .numpy()
            .tolist()
        )

        x_int8 = self.x
        if len(outlier_cols) > 0:
            x_fp = self.x[:, :, outlier_cols]
            w_fp = self.linear.weight[outlier_cols]
            res_fp = paddle.matmul(x_fp, w_fp)

            x_int8[:, :, outlier_cols] = 0
        x_int8, row_ranges = self.dynamic_quant(x_int8)

        res_int8 = paddle.matmul(x_int8, self.weight.transpose((1, 0)))
        dequant_scale = row_ranges * self.weight_scale / 127.0
        res_dequant = (res_int8.astype('float32') * dequant_scale).astype(
            self.dtype
        )

        if len(outlier_cols) > 0:
            out = res_dequant + res_fp
        else:
            out = res_dequant

        if self.bias:
            out += self.bias

        return out.numpy()

    def get_llm_int8_linear_out(self):
        out = Q.llm_int8_linear(
            self.x,
            self.weight,
            bias=self.linear.bias,
            weight_scale=self.weight_scale,
            threshold=self.threshold,
        )
        return out.numpy()

    def llm_int8_linear_out_static(self, out_expect):
        paddle.enable_static()
        main = paddle.static.Program()
        start = paddle.static.Program()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data("x", self.x.shape, dtype=self.dtype)

            weight = paddle.static.data(
                "weight", self.weight.shape, dtype='int8'
            )
            bias = paddle.static.data(
                "bias", self.linear.bias.shape, dtype=self.dtype
            )
            x_np = self.x.numpy()
            weight_np = self.weight.numpy()
            bias_np = self.linear.bias.numpy()
            if self.weight_scale is not None:
                weight_scale = paddle.static.data(
                    "weight_scale",
                    self.weight_scale.shape,
                    dtype='float32',
                )
                weight_scale_np = self.weight_scale.numpy()
            else:
                weight_scale = None
                weight_scale_np = None

            out = Q.llm_int8_linear(
                x,
                weight,
                bias,
                weight_scale,
                self.threshold,
            )
            feed_dict = {
                'x': x_np,
                'weight': weight_np,
                'bias': bias_np,
                "weight_scale": weight_scale_np,
            }
            exe = base.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out_real,) = exe.run(main, feed=feed_dict, fetch_list=[out])

        paddle.disable_static()

        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)

        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )

    def test_llm_int8_linear(self):
        out_expect = self.get_linear_out()
        if self.static:
            self.llm_int8_linear_out_static(out_expect)
            return
        else:
            out_real = self.get_llm_int8_linear_out()

        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)

        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase1(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int8"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase2(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int8"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class LLMInt8LinearTestCase4(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase5(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int4"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase7(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int8"
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase8(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int8"
        self.bias = False
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCase10(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.bias = False
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class LLMInt8LinearTestCaseStatic(LLMInt8LinearTestCase):
    def config(self):
        super().config()
        self.static = True


if __name__ == '__main__':
    unittest.main()
