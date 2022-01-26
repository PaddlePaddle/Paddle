#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import six
import abc
import unittest
import numpy as np
from scipy.special import expit, erf
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.tests.unittests.test_activation_op import TestActivation
from paddle.fluid.tests.unittests.test_gelu_op import gelu


@OpTestTool.skip_if_not_cpu_bf16()
@six.add_metaclass(abc.ABCMeta)
class MKLDNNBF16ActivationOp(object):
    @abc.abstractmethod
    def config(self):
        pass

    @abc.abstractmethod
    def op_forward(self, x):
        pass

    @abc.abstractmethod
    def op_grad(self, dout, x):
        pass

    def set_attrs(self):
        self.attrs = {"use_mkldnn": True}

    def init_data(self):
        self.x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)

    def setUp(self):
        self.dtype = np.uint16
        self.init_data()
        self.config()
        self.out = self.op_forward(self.x)

        self.inputs = {'X': convert_float_to_uint16(self.x)}
        self.outputs = {'Out': self.out}
        self.set_attrs()

    def calculate_grads(self):
        self.dx = self.op_grad(self.out, self.x)

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ["X"],
            "Out",
            user_defined_grads=[self.dx],
            user_defined_grad_outputs=[convert_float_to_uint16(self.out)])


class TestMKLDNNSigmoidBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "sigmoid"

    def op_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def op_grad(self, dout, x):
        return dout * self.op_forward(x) * (1 - self.op_forward(x))


class TestMKLDNNSqrtBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "sqrt"

    def init_data(self):
        self.x = np.random.uniform(1, 2, [2, 4, 3, 5]).astype(np.float32)

    def op_forward(self, x):
        return np.sqrt(x)

    def op_grad(self, dout, x):
        return dout / (2 * np.sqrt(x))


class TestMKLDNNGeluErfBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "gelu"

    def op_forward(self, x):
        return gelu(x, False)

    def op_grad(self, dout, x):
        return (dout *
                (0.5 + 0.5 * erf(x / np.sqrt(2)) +
                 (x / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.power(x, 2)))))


class TestMKLDNNGeluErfDim2BF16Op(TestMKLDNNGeluErfBF16Op):
    def init_data(self):
        self.x = np.random.uniform(-1, 1, [11, 17]).astype(np.float32)


class TestMKLDNNGeluTanhBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "gelu"

    def op_forward(self, x):
        return gelu(x, True)

    def op_grad(self, dout, x):
        grad_part = np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        return dout * 0.5 * (1 + grad_part) * (1 + np.sqrt(2 / np.pi) *
                                               (x + 0.134145 * np.power(x, 3)) *
                                               (1 - grad_part))

    def set_attrs(self):
        self.attrs = {"use_mkldnn": True, "approximate": True}


class TestMKLDNNGeluTanhDim2BF16Op(TestMKLDNNGeluTanhBF16Op):
    def init_data(self):
        self.x = np.random.uniform(-1, 1, [11, 17]).astype(np.float32)


class TestMKLDNNReluBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "relu"

    def op_forward(self, x):
        return np.maximum(x, 0)

    def op_grad(self, dout, x):
        return dout


class TestMKLDNNMishBF16Op(MKLDNNBF16ActivationOp, TestActivation):
    def config(self):
        self.op_type = "mish"

    def op_forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def op_grad(self, dout, x):
        omega = np.exp(3 * x) + 4 * np.exp(2 * x) + np.exp(x) * (4 * x + 6
                                                                 ) + 4 * (x + 1)
        delta = np.exp(2 * x) + 2 * np.exp(x) + 2
        return dout * ((np.exp(x) * omega) / delta**2)


if __name__ == '__main__':
    unittest.main()
