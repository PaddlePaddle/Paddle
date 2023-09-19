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

import os
import re
import struct
import unittest

import numpy as np

import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.base.framework import default_main_program
from paddle.framework import set_default_dtype

np.random.seed(123)
paddle.seed(123)
default_main_program().random_seed = 42


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.weight_dtype = "int8"
        self.static = False

    def setUp(self):
        self.config()
        if self.dtype == "bfloat16" or self.weight_dtype == "int4":
            self.atol = 1e-1
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

        self.bias = self.linear.bias
        self.weight = self.linear.weight
        self.weight_scale = None
        self.weight, self.weight_scale = Q.weight_quantize(
            self.weight,
            algo="weight_only_int8"
            if self.weight_dtype == "int8"
            else "weight_only_int4",
        )

    def get_linear_out(self):
        out = self.linear(self.x)
        return out.numpy()

    def get_weight_only_linear_out(self):
        out = Q.weight_only_linear(
            self.x,
            self.weight,
            bias=self.bias,
            weight_scale=self.weight_scale,
            weight_dtype=self.weight_dtype,
        )
        return out.numpy()

    def get_weight_only_linear_out_static(self):
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.program_guard(main, start):
            x = paddle.static.data("x", self.x.shape, dtype=self.x.dtype)

            weight = paddle.static.data(
                "weight", self.weight.shape, dtype=self.weight.dtype
            )
            bias = paddle.static.data(
                "bias", self.bias.shape, dtype=self.bias.dtype
            )
            x_np = self.x.numpy()
            weight_np = self.weight.numpy()
            bias_np = self.bias.numpy()
            if self.weight_scale is not None:
                weight_scale = paddle.static.data(
                    "weight_scale",
                    self.weight_scale.shape,
                    dtype=self.weight_scale.dtype,
                )
                weight_scale_np = self.weight_scale.numpy()
            else:
                weight_scale = None
                weight_scale_np = None

            out = Q.weight_only_linear(
                x,
                weight,
                bias,
                weight_scale,
                self.weight_dtype,
            )
            feed_dict = {
                'x': x_np,
                'weight': weight_np,
                'bias': bias_np,
                "weight_scale": weight_scale_np,
            }
            exe = base.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        paddle.disable_static()
        return out

    def test_weight_only_linear(self):
        out_expect = self.get_linear_out()
        if self.static:
            out_real = self.get_weight_only_linear_out_static()
        else:
            out_real = self.get_weight_only_linear_out()

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
class WeightOnlyLinearTestCase1(WeightOnlyLinearTestCase):
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
class WeightOnlyLinearTestCase2(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int8"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase3(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase4(WeightOnlyLinearTestCase):
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
class WeightOnlyLinearTestCase5(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int4"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase6(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase7(WeightOnlyLinearTestCase):
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
class WeightOnlyLinearTestCase8(WeightOnlyLinearTestCase):
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
class WeightOnlyLinearTestCase9(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase10(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
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
class WeightOnlyLinearTestCaseStatic(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.static = True


if __name__ == '__main__':
    unittest.main()
