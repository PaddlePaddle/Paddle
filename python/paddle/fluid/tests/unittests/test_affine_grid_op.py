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
from op_test import OpTest


def AffineGrid(theta, size):
    n = size[0]
    w = size[3]
    h = size[2]
    h_idx = np.repeat(
        np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[:, :, np.newaxis]
    w_idx = np.repeat(
        np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[:, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2)  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], size[0], axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

#    print ret.reshape([h * w, 2]).astype("float32")    
    return ret.reshape([n, h, w, 2]).astype("float32")


class TestAffineGridOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = "affine_grid"
        theta = np.random.randint(1, 3, self.theta_shape).astype("float32")
        theta = np.ones(self.theta_shape).astype("float32")
        self.inputs = {'Theta': theta}
        self.attrs = {"use_cudnn": True}
        if self.dynamic_shape:
            self.inputs['OutputShape'] = self.output_shape
        else:
            self.attrs['output_shape'] = self.output_shape
        self.outputs = {'Output': AffineGrid(theta, self.output_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['Theta'],
            'Output',
            no_grad_set=['OutputShape'],
            max_relative_error=0.006)

    def initTestCase(self):
        self.theta_shape = (3, 2, 3)
        self.output_shape = np.array([3, 2, 5, 7]).astype("int32")
        self.dynamic_shape = False


class TestAffineGridOpCase1(TestAffineGridOp):
    def initTestCase(self):
        self.theta_shape = (3, 2, 3)
        self.output_shape = np.array([3, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True


if __name__ == '__main__':
    unittest.main()
