#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, _set_use_system_allocator

import paddle
import paddle.nn.functional as F
from paddle.fluid import core

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
    # breakpoint()
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
        self.initConfig()
        self.initTestCase()
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output(
            no_check_set=None,
            atol=self.fw_comp_atol,
            rtol=self.fw_comp_rtol,
            check_prim=True,
            only_check_prim=True,
        )

    def test_check_grad_x(self):
        if self.data_format == "NCHW":
            self.check_grad(
                ["X"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
            )
        else:
            # origin batch_norm kernel differ in x_grad whether to calculate scale_grad and bias_grad
            self.check_grad_with_place(
                core.CPUPlace(),
                ["X"],
                ['Y'],
                user_defined_grad_outputs=self.out_grad,
                check_prim=True,
                only_check_prim=True,
            )

    def test_check_grad_scale_bias(self):
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        self.check_grad(
            ["X", "Scale", "Bias"],
            ['Y'],
            user_defined_grad_outputs=self.out_grad,
            check_prim=True,
            only_check_prim=True,
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
        np.random.seed(123)

        self.C = self.shape[1] if self.data_format == "NCHW" else self.shape[-1]
        x = np.random.random(self.shape).astype(self.dtype)
        weight = np.random.random(self.C).astype(self.dtype)
        bias = np.random.random(self.C).astype(self.dtype)
        running_mean = np.random.random(self.C).astype(self.dtype)
        running_var = np.random.random(self.C).astype(self.dtype)
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
        # y, mean, variance = _reference_batch_norm_naive(
        #     x, scale, bias, self.epsilon, self.begin_norm_axis
        # )
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
        # breakpoint()
        paddle.enable_static()
        self.outputs = {
            "Y": y,
            "MeanOut": running_mean,
            "VarianceOut": running_var,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance,
        }


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


class TestBatchNormOpNHWCShape1(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCShape2(TestBatchNormOp):
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
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCTestMode(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [2, 6, 2, 4]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
