#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np
from op_test import OpTest, _set_use_system_allocator, convert_float_to_uint16

import paddle
import paddle.nn.functional as F
from paddle.base import core

paddle.enable_static()

np.random.seed(123)
paddle.seed(123)

_set_use_system_allocator(True)


def batch_norm_wrapper(
    x,
    running_mean,
    running_variance,
    weight,
    bias,
    is_test,
    momentum,
    epsilon,
    data_format,
    use_global_stats,
):
    y = F.batch_norm(
        x,
        running_mean,
        running_variance,
        weight,
        bias,
        training=not is_test,
        momentum=momentum,
        epsilon=epsilon,
        data_format=data_format,
        use_global_stats=use_global_stats,
    )
    z = F.relu(y)
    return z


class TestBatchNormOp(OpTest):
    def setUp(self):
        self.python_api = batch_norm_wrapper
        self.public_python_api = batch_norm_wrapper
        self.op_type = "batch_norm"
        self.prim_op_type = "comp"
        self.python_out_sig = ["Y"]
        # (Todo: CZ) random error
        self.check_prim_pir = False
        self.check_prim_pir_grad = False
        self.check_cpu_prim_pir_grad = False

        self.initConfig()
        self.initTestCase()

    def test_check_output(self):
        if self.dtype not in ("uint16", "float16"):
            self.check_output_with_place(
                core.CPUPlace(),
                no_check_set=None,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_prim_pir,
            )
        if paddle.is_compiled_with_cuda():
            self.check_output_with_place(
                core.CUDAPlace(0),
                no_check_set=None,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_prim_pir,
            )

    def test_check_grad_x(self):
        if self.dtype not in ("uint16", "float16"):
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_cpu_prim_pir_grad,
            )
        if paddle.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0),
                ["X"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_prim_pir_grad,
            )

    def test_check_grad_scale_bias(self):
        if self.data_format == "NCHW" and self.training is False:
            self.enable_cinn = False
        if self.dtype == "float32":
            self.rev_comp_atol = 1e-3
            self.rev_comp_rtol = 1e-3
            self.cinn_atol = 1e-3
            self.cinn_rtol = 1e-3
        elif self.dtype == "float64":
            self.rev_comp_atol = 1e-12
            self.rev_comp_rtol = 1e-12
            self.cinn_atol = 1e-12
            self.cinn_rtol = 1e-12
        if self.dtype not in ("uint16", "float16"):
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X", "Scale", "Bias"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_cpu_prim_pir_grad,
            )
        if paddle.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0),
                ["X", "Scale", "Bias"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
                check_prim_pir=self.check_prim_pir_grad,
            )

    def initConfig(self):
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5

        self.cinn_atol = 1e-5
        self.cinn_rtol = 1e-5

        self.dtype = "float32"
        self.shape = [16, 24, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None

    def initTestCase(self):
        if (
            self.dtype in ("uint16", "float16")
            and not paddle.is_compiled_with_cuda()
        ):
            self.__class__.op_type = self.op_type
            self.__class__.no_need_check_grad = True
            return
        np.random.seed(123)

        self.C = self.shape[1] if self.data_format == "NCHW" else self.shape[-1]
        if self.dtype == "uint16":
            x = convert_float_to_uint16(
                np.random.random(self.shape).astype("float32")
            )
        else:
            x = np.random.random(self.shape).astype(self.dtype)

        self.var_dtype = (
            "float32" if self.dtype in ["float16", "uint16"] else self.dtype
        )
        weight = np.random.random(self.C).astype(self.var_dtype)
        bias = np.random.random(self.C).astype(self.var_dtype)
        running_mean = np.random.random(self.C).astype(self.var_dtype)
        running_var = np.random.random(self.C).astype(self.var_dtype)
        if self.dtype == "uint16":
            self.out_grad = [
                convert_float_to_uint16(
                    np.random.random(self.shape).astype("float32")
                )
            ]
        else:
            self.out_grad = [np.random.random(self.shape).astype(self.dtype)]
        self.inputs = {
            "X": x,
            "Scale": weight,
            "Bias": bias,
            "Mean": running_mean,
            "Variance": running_var,
        }

        if self.use_global_stats is None:
            self.use_global_stats = not self.training
            trainable_statistics = False
        else:
            trainable_statistics = not self.use_global_stats

        self.attrs = {
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "is_test": not self.training,
            "data_layout": self.data_format,
            "use_global_stats": self.use_global_stats,
            "trainable_statistics": trainable_statistics,
        }

        paddle.disable_static()

        (
            y,
            running_mean,
            running_var,
            saved_mean,
            saved_variance,
            _,
        ) = paddle._C_ops.batch_norm(
            paddle.to_tensor(x),
            paddle.to_tensor(running_mean),
            paddle.to_tensor(running_var),
            paddle.to_tensor(weight),
            paddle.to_tensor(bias),
            not self.training,
            self.momentum,
            self.epsilon,
            self.data_format,
            self.use_global_stats,
            trainable_statistics,
        )
        if self.dtype == "uint16":
            y = convert_float_to_uint16(y)
        paddle.enable_static()
        self.outputs = {
            "Y": y,
            "MeanOut": running_mean,
            "VarianceOut": running_var,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance,
        }


class TestBatchNormOpNCHWTestMode(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = True


class TestBatchNormOpNCHWFp64(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-11
        self.fw_comp_rtol = 1e-11
        self.rev_comp_atol = 1e-11
        self.rev_comp_rtol = 1e-11
        self.dtype = "float64"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None
        self.check_prim_pir = True
        self.check_cpu_prim_pir_grad = True
        self.check_prim_pir_grad = True


class TestBatchNormOpNCHWTestModeFp64(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-15
        self.fw_comp_rtol = 1e-15
        self.rev_comp_atol = 1e-15
        self.rev_comp_rtol = 1e-15
        self.dtype = "float64"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWFp16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        self.dtype = "float16"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWTestModeFp16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        self.dtype = "float16"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBatchNormOpNCHWbf16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        # prim bf16 has diff in windows
        if sys.platform == "win32":
            self.rev_comp_atol = 5e-3
            self.rev_comp_rtol = 5e-3
        self.cinn_atol = 1e-3
        self.cinn_rtol = 1e-3
        self.dtype = "uint16"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None
        # Todo(CZ): open this
        self.check_prim_pir = False
        self.check_cpu_prim_pir_grad = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBatchNormOpNCHWTestModebf16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        # prim bf16 has diff in windows
        if sys.platform == "win32":
            self.rev_comp_atol = 5e-3
            self.rev_comp_rtol = 5e-3
        self.cinn_atol = 1e-3
        self.cinn_rtol = 1e-3
        self.dtype = "uint16"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWShape2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 8, 16, 32]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWMomentum2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWEps2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-06
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWShape3(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 8, 32]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


class TestBatchNormOpNCHWShape4(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 256]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = None


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
