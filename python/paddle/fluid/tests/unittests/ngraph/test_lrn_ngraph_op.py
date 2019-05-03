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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest


class TestLRNOp(OpTest):
    def get_input(self):
        x = np.random.rand(self.N, self.C, self.H, self.W).astype("float32")
        return x + 1

    def get_out(self):
        start = -(self.n - 1) // 2
        end = start + self.n

        mid = np.empty((self.N, self.C, self.H, self.W)).astype("float32")
        mid.fill(self.k)
        for m in range(0, self.N):
            for i in range(0, self.C):
                for c in range(start, end):
                    ch = i + c
                    if ch < 0 or ch >= self.C:
                        continue

                    s = mid[m][i][:][:]
                    r = self.x[m][ch][:][:]
                    s += np.square(r) * self.alpha

        mid2 = np.power(mid, -self.beta)
        return np.multiply(self.x, mid2), mid

    def get_attrs(self):
        attrs = {
            'n': self.n,
            'k': self.k,
            'alpha': self.alpha,
            'beta': self.beta
        }

    def setUp(self):
        self.op_type = "lrn"
        self.N = 2
        self.C = 3
        self.H = 5
        self.W = 5

        self.n = 5
        self.k = 2.
        self.alpha = 0.0001
        self.beta = 0.75
        self.x = self.get_input()
        self.out, self.mid_out = self.get_out()

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out, 'MidOut': self.mid_out}
        self.attrs = self.get_attrs()

    def test_check_output(self):
        self.check_output(atol=0.002)


if __name__ == '__main__':
    unittest.main()
