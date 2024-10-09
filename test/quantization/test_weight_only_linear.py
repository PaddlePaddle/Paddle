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

import copy
import math
import os
import re
import struct
import unittest

import numpy as np

import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.framework import set_default_dtype
from paddle.pir_utils import IrGuard

np.random.seed(123)
paddle.seed(123)


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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
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
        self.group_size = -1

    def weightQuantizeCPUGPUConsistenceCheck(self, weight_float):
        for arch in [70, 75, 80, 86]:
            weight_gpu, weight_scale_gpu = Q.weight_quantize(
                (
                    weight_float.cuda()
                    if self.weight_dtype == "int8"
                    else self.weight.cpu()
                ),
                algo=(
                    "weight_only_int8"
                    if self.weight_dtype == "int8"
                    else "weight_only_int4"
                ),
                arch=arch,
                group_size=self.group_size,
            )
            weight_cpu, weight_scale_cpu = Q.weight_quantize(
                weight_float.cpu(),
                algo=(
                    "weight_only_int8"
                    if self.weight_dtype == "int8"
                    else "weight_only_int4"
                ),
                arch=arch,
                group_size=self.group_size,
            )
            np.testing.assert_allclose(
                weight_gpu.numpy(),
                weight_cpu.numpy(),
                atol=1.5,
                rtol=2,
            )
            np.testing.assert_allclose(
                weight_scale_gpu.numpy(),
                weight_scale_cpu.numpy(),
                atol=1e-5,
                rtol=1e-3,
            )
            pass
        pass

    def setUp(self):
        self.config()
        if self.dtype == "bfloat16" or self.weight_dtype == "int4":
            self.atol = 1.3e-1
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
        self.float_weight = self.linear.weight
        self.weight_scale = None
        # check weight quantize
        self.weightQuantizeCPUGPUConsistenceCheck(self.float_weight)

        self.weight, self.weight_scale = Q.weight_quantize(
            (
                self.float_weight.cuda()
                if self.weight_dtype == "int8"
                else self.weight.cpu()
            ),
            algo=(
                "weight_only_int8"
                if self.weight_dtype == "int8"
                else "weight_only_int4"
            ),
            group_size=self.group_size,
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
            group_size=self.group_size,
        )
        return out.numpy()

    def test_weight_only_linear(self):
        out_expect = self.get_linear_out()
        out_real = self.get_weight_only_linear_out()

        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase1(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int8"


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
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
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase11(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase12(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase13(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase14(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase15(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 64


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase16(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul groupwise mode need CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase17(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 64


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul groupwise mode need CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase18(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase19(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 2
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase20(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 64


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class WeightOnlyLinearTestCase21(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase22(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int8"
        self.in_features = 128
        self.out_features = 288


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase23(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int8"
        self.in_features = 128
        self.out_features = 288


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase24(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.in_features = 128
        self.out_features = 288


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase25(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase26(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.group_size = 64


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase27(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase28(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int4"
        self.token = 300
        self.group_size = 128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearTestCase29(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = "int8"
        self.token = 300
        self.group_size = 128


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

    def get_weight_only_linear_out_static(self):
        paddle.enable_static()
        main = paddle.static.Program()
        start = paddle.static.Program()
        with paddle.static.program_guard(main, start):
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
                group_size=self.group_size,
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

    def test_weight_quantize_and_dequantize_pir(self, algo='weight_only_int8'):
        with IrGuard():
            weight = (
                paddle.rand(shape=(4096, 12288), dtype='float16')
                * 1
                / math.sqrt(4096)
            )

            quant_weight, quant_scale = Q.weight_quantize(x=weight, algo=algo)
            dequant_weight = Q.weight_dequantize(
                quant_weight, quant_scale, algo=algo
            )
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            res = exe.run(feed={}, fetch_list=[weight, dequant_weight])
            np.testing.assert_allclose(res[0], res[1], rtol=1e-2, atol=1e-2)

    def test_weight_quantize_and_dequantize_int4_pir(
        self, algo='weight_only_int4'
    ):
        with IrGuard():
            weight = (
                paddle.rand(shape=(4096, 12288), dtype='float16')
                * 1
                / math.sqrt(4096)
            )
            quant_weight, quant_scale = Q.weight_quantize(x=weight, algo=algo)
            dequant_weight = Q.weight_dequantize(
                quant_weight, quant_scale, algo=algo
            )
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            res = exe.run(feed={}, fetch_list=[weight, dequant_weight])
            np.testing.assert_allclose(res[0], res[1], rtol=1e-1, atol=1e-1)

    def test_weight_only_linear(self):
        out_expect = self.get_linear_out()

        out_real = self.get_weight_only_linear_out_static()
        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )

        with IrGuard():
            out_real = self.get_weight_only_linear_out_static()
        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyQuantizeCPUGPUTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.group_size = -1

    def weightQuantizeCPUGPUConsistenceCheck(self, weight_float):
        for arch in [70, 75, 80, 86]:
            weight_gpu, weight_scale_gpu = Q.weight_quantize(
                weight_float.cuda(),
                algo="weight_only_int4",
                arch=arch,
                group_size=self.group_size,
            )
            weight_cpu, weight_scale_cpu = Q.weight_quantize(
                weight_float.cpu(),
                algo="weight_only_int4",
                arch=arch,
                group_size=self.group_size,
            )
            np.testing.assert_allclose(
                weight_gpu.numpy(),
                weight_cpu.numpy(),
                atol=17,
            )
            np.testing.assert_allclose(
                weight_scale_gpu.numpy(),
                weight_scale_cpu.numpy(),
                atol=1e-5,
                rtol=1e-3,
            )

    def setUp(self):
        self.config()
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        set_default_dtype(self.dtype)
        if self.bias:
            bias_attr = base.ParamAttr(
                trainable=False,
                regularizer=None,
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
        else:
            bias_attr = None
        self.linear = paddle.nn.Linear(
            self.in_features, self.out_features, bias_attr=bias_attr
        )

        self.bias = self.linear.bias
        self.float_weight = self.linear.weight
        self.weightQuantizeCPUGPUConsistenceCheck(self.float_weight)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class WeightOnlyLinearBackwardAndWeightDequantizeTestCase(unittest.TestCase):
    def test_weightonly_linear_backward(
        self, algo='weight_only_int8', weight_dtype='int8'
    ):
        x = (
            paddle.rand(shape=(128, 4096), dtype='float16')
            * 1
            / math.sqrt(4096)
        )
        x.stop_gradient = False
        quant_x = copy.deepcopy(x)
        quant_x.stop_gradient = False
        weight = (
            paddle.rand(shape=(4096, 12288), dtype='float16')
            * 1
            / math.sqrt(4096)
        )

        quant_weight, quant_scale = Q.weight_quantize(
            x=weight.cuda(), algo=algo
        )
        dequant_weight = Q.weight_dequantize(
            quant_weight.cuda(), quant_scale, algo=algo
        )
        np.testing.assert_allclose(weight, dequant_weight, rtol=1e-2, atol=1e-2)

        quant_out = Q.weight_only_linear(
            x=quant_x,
            weight=quant_weight,
            weight_scale=quant_scale,
            weight_dtype=weight_dtype,
        )
        out = paddle.matmul(x=x, y=weight)
        np.testing.assert_allclose(quant_out, out, rtol=1e-2, atol=1e-2)

        quant_out.backward()
        out.backward()
        np.testing.assert_allclose(quant_x.grad, x.grad, rtol=1e-2, atol=1e-2)

    def test_weightonly_linear_backward_int4(self):
        def test_weightonly_linear_backward(
            self, algo='weight_only_int4', weight_dtype='int4'
        ):
            x = (
                paddle.rand(shape=(128, 4096), dtype='float16')
                * 1
                / math.sqrt(4096)
            )
            x.stop_gradient = False
            quant_x = copy.deepcopy(x)
            quant_x.stop_gradient = False
            weight = (
                paddle.rand(shape=(4096, 12288), dtype='float16')
                * 1
                / math.sqrt(4096)
            )

            quant_weight, quant_scale = Q.weight_quantize(
                x=weight.cuda(), algo=algo
            )
            quant_weight = quant_weight.view(
                [quant_weight.shape[0] * 2, quant_weight.shape[1] // 2]
            )
            dequant_weight = Q.weight_dequantize(
                quant_weight.cuda(), quant_scale, algo=algo
            )
            np.testing.assert_allclose(
                weight, dequant_weight, rtol=1e-2, atol=1e-2
            )

            quant_out = Q.weight_only_linear(
                x=quant_x,
                weight=quant_weight,
                weight_scale=quant_scale,
                weight_dtype=weight_dtype,
            )
            out = paddle.matmul(x=x, y=weight)
            np.testing.assert_allclose(quant_out, out, rtol=1e-3, atol=1e-3)

            quant_out.backward()
            out.backward()
            np.testing.assert_allclose(
                quant_x.grad, x.grad, rtol=1e-3, atol=1e-3
            )


if __name__ == '__main__':
    unittest.main()
