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


def affinity_propagate(x, gate_weight, mask=None, kernel_size=3):
    shape = x.shape
    side_num = int((kernel_size - 1) / 2)
    if len(x.shape) == 4:
        xs = []
        for i in range(2 * side_num + 1):
            for j in range(2 * side_num + 1):
                if i != side_num or j != side_num:
                    rx = np.pad(x, ((0, 0), (0, 0), (i, 2 * side_num - i),
                                    (j, 2 * side_num - j)), 'constant')
                    xs.append(rx[:, np.newaxis, :, :, :])
        expand_x = np.concatenate(xs, axis=1)[:, :, :, 1:-1, 1:-1]
        out = np.sum(expand_x * gate_weight[:, :, np.newaxis, :, :], axis=1)
        gate_sum = np.sum(gate_weight, axis=1, keepdims=True)
        out = (1.0 - gate_sum) * x + out

        if mask is not None:
            mask_sign = np.sign(mask)
            out = (1. - mask_sign) * out + mask_sign * x
        return out


class TestAffinityPropagateOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'affinity_propagate'

        np.random.seed(2333)
        gate_weight = np.random.uniform(
            -1, 1, self.gate_weight_shape).astype('float32')
        x = np.random.uniform(-1, 1, self.x_shape).astype('float32')
        mask = np.random.uniform(-1, 1, self.x_shape).astype('float32')

        self.attrs = {"kernel_size": self.kernel_size, }

        self.inputs = {
            'X': x,
            'GateWeight': gate_weight,
            'Mask': mask,
        }

        output = affinity_propagate(x, gate_weight, mask)

        self.outputs = {'Out': output, }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    # def test_check_grad_ignore_gtbox(self):
    #     place = core.CPUPlace()
    #     self.check_grad_with_place(place, ['X'], 'Loss', max_relative_error=0.2)

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 5)
        self.gate_weight_shape = (2, 8, 5, 5)
        self.kernel_size = 3


if __name__ == "__main__":
    unittest.main()
