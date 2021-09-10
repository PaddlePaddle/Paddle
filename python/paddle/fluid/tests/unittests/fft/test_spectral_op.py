# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import sys.path
from ..op_test import OpTest
from paddle.fluid import Program, program_guard
import paddle.fluid.dygraph as dg
import paddle.static as static
from numpy.random import random as rand

paddle.enable_static()


class TestFFTC2ROp(OpTest):
    def setUp(self):
        self.op_type = "fft_c2r"
        self.init_dtype_type()
        self.init_input_output()
        self.init_grad_input_output()

    def init_dtype_type(self):
        self.dtype = np.complex64

    def init_input_output(self):
        x = (np.random.random((12, 14)) + 1j * np.random.random(
            (12, 14))).astype(self.dtype)
        out = np.conj(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_grad_input_output(self):
        self.grad_out = (np.ones((12, 14)) + 1j * np.ones(
            (12, 14))).astype(self.dtype)
        self.grad_in = np.conj(self.grad_out)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.grad_in],
            user_defined_grad_outputs=[self.grad_out])
