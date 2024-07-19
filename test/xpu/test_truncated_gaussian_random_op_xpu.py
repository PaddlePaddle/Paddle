#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest
from scipy.special import erf

import paddle
from paddle import base
from paddle.base.executor import Executor

paddle.enable_static()


def normal_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def normal_pdf(x):
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


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


class XPUTestTruncatedGaussianRandomOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'truncated_gaussian_random'
        self.use_dynamic_create_class = False

    class TestTruncatedGaussianRandomOp(XPUOpTest):
        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
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
            self.gaussian_random_test(place=base.XPUPlace(0))

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


support_types = get_xpu_op_support_types('truncated_gaussian_random')
for stype in support_types:
    create_test_class(globals(), XPUTestTruncatedGaussianRandomOp, stype)

if __name__ == "__main__":
    unittest.main()
