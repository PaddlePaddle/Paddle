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
from test_sparse_attention_op import get_cuda_version

import paddle
import paddle.incubate.nn.functional as F
from paddle import fluid
from paddle.fluid import core
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


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
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


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase1(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase2(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class QuantizedMatmulTestCase3(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase4(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase5(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.bias = False


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class QuantizedMatmulTestCase6(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase7(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int4"
        self.atol = 1e-1


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase8(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int4"
        self.bias = False
        self.atol = 1e-1


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class QuantizedMatmulTestCase9(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int4"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase10(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase11(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1
        self.bias = False


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class QuantizedMatmulTestCase12(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "weight_only_int8"
        self.batch = 1
        self.token = 1


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase13(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "llm.int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCase14(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.quant_method = "llm.int8"
        self.bias = False


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16",
)
class QuantizedMatmulTestCase15(QuantizedMatmulTestCase):
    def config(self):
        super().config()
        self.dtype = 'bfloat16'
        self.quant_method = "llm.int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCaseStatic(unittest.TestCase):
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
        paddle.disable_static()
        self.config()
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
        paddle.enable_static()
        main = fluid.Program()
        start = fluid.Program()
        with fluid.program_guard(main, start):
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

            out = F.quantized_matmul(
                x, weight, bias, weight_scale, self.quant_method
            )
            feed_dict = {
                'x': x_np,
                'weight': weight_np,
                'bias': bias_np,
                "weight_scale": weight_scale_np,
            }
            exe = fluid.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        paddle.disable_static()
        return out

    def test_quantized_matmul(self):
        out_real = self.get_quantized_matmul_out()
        out_expect = self.get_linear_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCaseStatic1(QuantizedMatmulTestCaseStatic):
    def config(self):
        super().config()
        self.bias = False
        self.quant_method = "weight_only_int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCaseStatic2(QuantizedMatmulTestCaseStatic):
    def config(self):
        super().config()
        self.quant_method = "weight_only_int4"
        self.atol = 1e-1


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestCaseStatic3(QuantizedMatmulTestCaseStatic):
    def config(self):
        super().config()
        self.quant_method = "llm.int8"


@unittest.skipIf(
    get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class QuantizedMatmulTestError(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.quant_method = "None"

    def setUp(self):
        self.config()
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(self.in_features, self.out_features)

        self.weight = self.linear.weight
        self.weight_scale = None

    def test_errors(self):
        def dynamic_quant_method():
            out = F.quantized_matmul(
                self.x,
                self.weight,
                quant_method="abc",
            )

        self.assertRaises(ValueError, dynamic_quant_method)

        def static_quant_method():
            paddle.enable_static()
            main = fluid.Program()
            start = fluid.Program()
            with fluid.program_guard(main, start):
                x = paddle.static.data("x", self.x.shape, dtype=self.x.dtype)
                weight = paddle.static.data(
                    "weight", self.weight.shape, dtype=self.weight.dtype
                )

                out = F.quantized_matmul(x, weight, quant_method='abc')

        self.assertRaises(ValueError, static_quant_method)


if __name__ == '__main__':
    unittest.main()
