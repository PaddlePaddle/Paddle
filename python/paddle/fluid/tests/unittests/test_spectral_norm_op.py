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
from op_test import OpTest

from paddle.fluid import core


def spectral_norm(weight, u, v, dim, power_iters, eps):
    shape = weight.shape
    weight_mat = weight.copy()
    h = shape[dim]
    w = np.prod(shape) // h
    if dim != 0:
        perm = [dim] + [d for d in range(len(shape)) if d != dim]
        weight_mat = weight_mat.transpose(perm)
    weight_mat = weight_mat.reshape((h, w))

    u = u.reshape((h, 1))
    v = v.reshape((w, 1))
    for i in range(power_iters):
        v = np.matmul(weight_mat.T, u)
        v_norm = np.sqrt((v * v).sum())
        v = v / (v_norm + eps)
        u = np.matmul(weight_mat, v)
        u_norm = np.sqrt((u * u).sum())
        u = u / (u_norm + eps)

    sigma = (u * np.matmul(weight_mat, v)).sum()
    return weight / sigma


class TestSpectralNormOpNoGrad(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'spectral_norm'
        weight = np.random.random(self.weight_shape).astype('float32')
        u = np.random.normal(0., 1., self.u_shape).astype('float32')
        v = np.random.normal(0., 1., self.v_shape).astype('float32')

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

        output = spectral_norm(weight, u, v, self.dim, self.power_iters,
                               self.eps)
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.weight_shape = (2, 3)
        self.u_shape = (2, )
        self.v_shape = (3, )
        self.dim = 0
        self.power_iters = 5
        self.eps = 1e-12


class TestSpectralNormOpNoGrad2(TestSpectralNormOpNoGrad):
    def initTestCase(self):
        self.weight_shape = (2, 3, 3, 3)
        self.u_shape = (3, )
        self.v_shape = (18, )
        self.dim = 1
        self.power_iters = 10
        self.eps = 1e-12


class TestSpectralNormOp(TestSpectralNormOpNoGrad):
    def test_check_grad_ignore_uv(self):
        self.check_grad(
            ['Weight'],
            'Out',
            no_grad_set=set(["U", "V"]),
            max_relative_error=0.1)

    def initTestCase(self):
        self.weight_shape = (2, 3)
        self.u_shape = (2, )
        self.v_shape = (3, )
        self.dim = 0
        self.power_iters = 0
        self.eps = 1e-12


class TestSpectralNormOp2(TestSpectralNormOp):
    def initTestCase(self):
        self.weight_shape = (2, 3, 3, 3)
        self.u_shape = (3, )
        self.v_shape = (18, )
        self.dim = 1
        self.power_iters = 0
        self.eps = 1e-12


if __name__ == "__main__":
    unittest.main()
