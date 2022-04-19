#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

class TestEinsumBinary(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "einsum"
        self.disable = False
        self.set_mandatory()
        self.init_input()
        np.random.seed(123)
        out = np.einsum(self.equation, *self.inputs)
        operands = []
        for idx, inp in enumerate(self.inputs):
            operands.append(("x" + str(idx), inp))
        self.inputs = {"Operands": operands}
        self.attrs = {"equation": self.equation}
        self.outputs = {'Out': out}

    def init_input(self):
        self.inputs = []
        for t, s in zip(self.types, self.shapes):
            self.inputs.append(
                np.random.random(s).astype(t))

    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "imjl,jklm->imk"

    def test_check_output(self):
        if not self.disable: 
            self.check_output()

    #def test_grad(self):
        #self.check_grad(["X"], ["Eigenvalues"])

class TestEinsum1(TestEinsumBinary):# {{{
    def set_mandatory(self):
        self.shapes = [(10, 3, 3), (10, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->mik"

class TestEinsum2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 3), (10, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->ikm"

class TestEinsum3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 3), (3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "ij,jk->ik"# }}}
        
class TestEinsumWithReduction(TestEinsumBinary):# {{{
    def set_mandatory(self):
        self.shapes = [(10, 3, 5), (5, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "ijk, kl->jl"
#}}}

class TestEinsumWithReduction1(TestEinsumBinary):# {{{
    def set_mandatory(self):
        self.shapes = [(10, 10, 3, 5), (10, 5, 10, 100)]
        self.types = [np.float64, np.float64]
        self.equation = "mijk, mklh->ljm"
#}}}

class TestEinsumWithUnary(TestEinsumBinary):# {{{
    def set_mandatory(self):
        self.shapes = [(10, 10, 3, 5)]
        self.types = [np.float64]
        self.equation = "mijk->mi"
#}}}

class TestEinsumWithUnary1(TestEinsumBinary):# {{{
    def set_mandatory(self):
        self.shapes = [(10, 3, 5)]
        self.types = [np.float64]
        self.equation = "mij->jmi"
#}}}

if __name__ == "__main__":
    unittest.main()

