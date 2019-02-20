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

from __future__ import division

import unittest
import numpy as np
from op_test import OpTest

from paddle.fluid import core


class TestSpectralNormOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'spectral_norm'
        # weight = np.random.random(self.weight_shape).astype('float32')
        # u = np.random.random(self.u_shape).astype('float32')
        # v = np.random.random(self.u_shape).astype('float32')
        weight = np.ones(self.weight_shape).astype('float32')
        weight[1, :] = 2.
        u = np.ones(self.u_shape).astype('float32')
        v = np.ones(self.v_shape).astype('float32')

        self.attrs = {
            "dim": self.dim,
            "power_iters": self.power_iters,
            "eps": self.eps,
        }

        self.inputs = {
            "Weight": weight,
            "U": u,
            "V": v,
        }

        output = weight
        self.outputs = {"Out": weight, }

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.weight_shape = (2, 3)
        self.u_shape = (2, )
        self.v_shape = (3, )
        self.dim = 0
        self.power_iters = 1
        self.eps = 1e-12


if __name__ == "__main__":
    unittest.main()
