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

import re
import sys
import unittest

import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid.dygraph as dg
import paddle.static as static
import scipy.stats
from numpy.random import random as rand
sys.path.append("../")
from op_test import OpTest
from paddle.fluid import Program, program_guard

paddle.enable_static()


class TestDirichletOp(OpTest):
    # Because dirichlet random sample have not gradient, we skip gradient check.
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "dirichlet"
        self.alpha = np.array((1., 2.))
        self.sample_shape = (100000, 2)

        self.inputs = {'Alpha': np.broadcast_to(self.alpha, self.sample_shape)}
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape)}

    def test_check_output(self):
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(
            scipy.stats.kstest(
                outs[0][:, 0],
                # scipy dirichlet have not cdf, use beta to replace it.
                scipy.stats.beta(
                    a=self.alpha[0], b=self.alpha[1]).cdf)[0],
            0.01)
