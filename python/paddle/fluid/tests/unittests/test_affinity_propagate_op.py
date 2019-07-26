#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import logit
from scipy.special import expit
from op_test import OpTest

from paddle.fluid import core


class TestAffinityPropagateOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'affinity_propagate'

        np.random.seed(2333)
        guidance = np.random.uniform(-1, 1, self.guidance_shape).astype('float32')
        x = np.random.uniform(-1, 1, self.x_shape).astype('float32')
        print(guidance)

        self.attrs = {
            "prop_iters": self.prop_iters,
            "kernel_size": self.kernel_size,
            "norm_type": self.norm_type,
        }

        self.inputs = {
            'X': x,
            'Guidance': guidance,
        }

        output = x.copy()

        self.outputs = {
            'Out': output,
        }

    # def test_check_output(self):
    #     self.check_output()

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-3)

    # def test_check_grad_ignore_gtbox(self):
    #     place = core.CPUPlace()
    #     self.check_grad_with_place(place, ['X'], 'Loss', max_relative_error=0.2)

    def initTestCase(self):
        self.x_shape = (1, 1, 2, 2)
        self.guidance_shape = (1, 8, 2, 2)
        self.prop_iters = 1
        self.kernel_size = 3
        self.norm_type = 'sum'


if __name__ == "__main__":
    unittest.main()
