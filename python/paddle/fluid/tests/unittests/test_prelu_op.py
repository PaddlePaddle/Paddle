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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest


class PReluTest(OpTest):
    def setUp(self):
        print('setUp')
        import sys
        sys.stdout.flush()
        self.op_type = "prelu"
        self.initTestCase()
        x_np = np.random.normal(size=(3, 5, 5, 10)).astype("float32")

        # Since zero point in prelu is not differentiable, avoid randomize
        # zero.
        x_np[np.abs(x_np) < 0.005] = 0.02

        if self.attrs == {'mode': "all"}:
            alpha_np = np.random.rand(1).astype("float32")
            self.inputs = {'X': x_np, 'Alpha': alpha_np}
        elif self.attrs == {'mode': "channel"}:
            alpha_np = np.random.rand(1, x_np.shape[1], 1, 1).astype("float32")
            self.inputs = {'X': x_np, 'Alpha': alpha_np}
        else:
            alpha_np = np.random.rand(*x_np.shape).astype("float32")
            self.inputs = {'X': x_np, 'Alpha': alpha_np}

        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'],
                                     0.) * self.inputs['Alpha']
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def tearDown(self):
        print('tearDown')
        import sys
        sys.stdout.flush()
        del self.outputs
        del self.inputs

    def initTestCase(self):
        self.attrs = {'mode': "channel"}

    def test_check_4_output(self):
        print('test_check_0_output')
        import sys
        sys.stdout.flush()
        self.check_output()

    def test_check_0_grad_2_ignore_x(self):
        print('test_check_2_grad_2_ignore_x')
        import sys
        sys.stdout.flush()
        self.check_grad(['Alpha'], 'Out', no_grad_set=set('X'))

    # TODO(minqiyang): remove the order of tests
    def test_check_1_grad_1(self):
        print('test_check_1_grad_1')
        import sys
        sys.stdout.flush()
        self.check_grad(['X', 'Alpha'], 'Out')

    def test_check_3_grad_3_ignore_alpha(self):
        print('test_check_3_grad_3_ignore_alpha')
        import sys
        sys.stdout.flush()
        self.check_grad(['X'], 'Out', no_grad_set=set('Alpha'))


class TestCase1(PReluTest):
    def initTestCase(self):
        self.attrs = {'mode': "all"}


#class TestCase2(PReluTest):
#    def initTestCase(self):
#        self.attrs = {'mode': "channel"}
#
#
#class TestCase3(PReluTest):
#    def initTestCase(self):
#        self.attrs = {'mode': "element"}

if __name__ == "__main__":
    unittest.main()
