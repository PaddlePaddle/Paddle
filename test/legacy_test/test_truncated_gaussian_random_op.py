#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
from op_test import OpTest, convert_uint16_to_float, get_device_place

import paddle
from paddle import base
from paddle.base import core
from paddle.base.executor import Executor

paddle.enable_static()


def normal_cdf(x):
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pdf(x):
    """Probability density function for standard normal distribution."""
    return math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def truncated_normal_mean(mean, std, a, b):
    '''Reference: https://en.wikipedia.org/wiki/Truncated_normal_distribution'''
    alpha = (a - mean) / std
    beta = (b - mean) / std
    z = normal_cdf(beta) - normal_cdf(alpha)
    return mean + (normal_pdf(alpha) - normal_pdf(beta)) / z * std


def truncated_normal_var(mean, std, a, b):
    '''Reference: https://en.wikipedia.org/wiki/Truncated_normal_distribution'''
    alpha = (a - mean) / std
    beta = (b - mean) / std
    z = normal_cdf(beta) - normal_cdf(alpha)
    return std**2 * (
        1
        - (beta * normal_pdf(beta) - alpha * normal_pdf(alpha)) / z
        - ((normal_pdf(alpha) - normal_pdf(beta)) / z) ** 2
    )


class TestTruncatedGaussianRandomOp(OpTest):
    def init(self):
        self.dtype = np.float32
        self.place = get_device_place()
        self.__class__.op_type = "truncated_gaussian_random"

    def setUp(self):
        self.init()
        self.inputs = {}
        self.set_attrs()
        self.attrs = {
            "shape": self.shape,
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "a": self.a,
            "b": self.b,
        }
        self.outputs = {'Out': np.zeros(self.shape, dtype=self.dtype)}

    def set_attrs(self):
        self.shape = [10000]
        self.mean = 0.0
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.gaussian_random_test(place=base.CUDAPlace(0))
        else:
            self.gaussian_random_test(place=base.CPUPlace())

    def gaussian_random_test(self, place):
        with paddle.pir_utils.OldIrGuard():
            program = base.Program()
            block = program.global_block()
            vout = block.create_var(name="Out")
            op = block.append_op(
                type=self.op_type, outputs={"Out": vout}, attrs=self.attrs
            )

            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)

            fetch_list = []
            for var_name in self.outputs:
                fetch_list.append(block.var(var_name))

            exe = Executor(place)
            outs = exe.run(program, fetch_list=fetch_list)
            tensor = outs[0]
            np.testing.assert_allclose(
                np.mean(tensor),
                truncated_normal_mean(self.mean, self.std, self.a, self.b),
                atol=0.05,
            )
            np.testing.assert_allclose(
                np.var(tensor),
                truncated_normal_var(self.mean, self.std, self.a, self.b),
                atol=0.05,
            )


class TestTruncatedGaussianRandomOp_1(TestTruncatedGaussianRandomOp):
    def set_attrs(self):
        self.shape = [4096, 2]
        self.mean = 5.0
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0


class TestTruncatedGaussianRandomOp_2(TestTruncatedGaussianRandomOp):
    def set_attrs(self):
        self.shape = [1024]
        self.mean = -2.0
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0


class TestTruncatedGaussianRandomOp_3(TestTruncatedGaussianRandomOp):
    def set_attrs(self):
        self.shape = [11 * 13 * 17]
        self.mean = -1.0
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0


class TestTruncatedGaussianRandomOp_4(TestTruncatedGaussianRandomOp):
    def set_attrs(self):
        self.shape = [2049]
        self.mean = 5.1234
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0


class TestTruncatedGaussianRandomOpFp64(TestTruncatedGaussianRandomOp):
    def init(self):
        self.dtype = np.float64
        self.place = get_device_place()
        self.__class__.op_type = "truncated_gaussian_random"


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestTruncatedGaussianRandomOpBf16(TestTruncatedGaussianRandomOp):
    def init(self):
        self.dtype = np.uint16  # bfloat16 is represented as uint16 in numpy
        self.place = get_device_place()
        self.__class__.op_type = "truncated_gaussian_random"

    def set_attrs(self):
        self.shape = [10000]
        self.mean = 0.0
        self.std = 1.0
        self.a = -2.0
        self.b = 2.0

    def gaussian_random_test(self, place):
        from paddle.base import core

        with paddle.pir_utils.OldIrGuard():
            program = base.Program()
            block = program.global_block()
            vout = block.create_var(name="Out")
            # For bfloat16, need to specify dtype in attrs
            attrs_with_dtype = {
                **self.attrs,
                "dtype": core.VarDesc.VarType.BF16,
            }
            op = block.append_op(
                type=self.op_type, outputs={"Out": vout}, attrs=attrs_with_dtype
            )

            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)

            fetch_list = []
            for var_name in self.outputs:
                fetch_list.append(block.var(var_name))

            exe = Executor(place)
            outs = exe.run(program, fetch_list=fetch_list)
            # bfloat16 output needs to be converted to float32 for verification
            tensor = convert_uint16_to_float(outs[0])
            np.testing.assert_allclose(
                np.mean(tensor),
                truncated_normal_mean(self.mean, self.std, self.a, self.b),
                atol=0.1,  # Relaxed tolerance due to lower precision of bfloat16
            )
            np.testing.assert_allclose(
                np.var(tensor),
                truncated_normal_var(self.mean, self.std, self.a, self.b),
                atol=0.15,  # Relaxed tolerance due to lower precision of bfloat16
            )


if __name__ == "__main__":
    unittest.main()
