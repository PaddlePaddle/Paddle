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
        self.operands = []
        for idx, inp in enumerate(self.inputs):
            self.operands.append(("x" + str(idx), inp))
        self.inputs = {"Operands": self.operands}
        self.attrs = {"equation": self.equation}
        self.outputs = {'Out': out}

    def init_input(self):
        self.inputs = []
        for t, s in zip(self.types, self.shapes):
            self.inputs.append(np.random.random(s).astype(t))

    def set_mandatory(self):
        self.disable = False
        self.shapes = [(10, 10, 20), (20, 6)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,jk->ki"

    def test_check_output(self):
        if not self.disable:
            self.check_output()

    def test_grad(self):
        if not self.disable:
            self.check_grad([op[0] for op in self.operands], ["Out"])


class TestEinsum1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->mik"


class TestEinsum2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->ikm"


class TestEinsum3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10), (10, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "ij,jk->ik"  # }}}


class TestEinsumWithReduction(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 5), (5, 30)]
        self.types = [np.float64, np.float64]
        self.equation = "ijk,kl->jl"


class TestEinsumWithReduction1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 3, 5), (10, 5, 10, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "mijk,mklh->ljm"


class TestEinsumWithUnary(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10, 3, 5)]
        self.types = [np.float64]
        self.equation = "mijk->mi"


class TestEinsumWithUnary1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "imjl,jklm->imk"


class TestEinsumWithBroadcast1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3)]
        self.types = [np.float64]
        self.equation = "i...->..."


class TestEinsumWithBroadcast2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 11), (3, 4, 5, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "...ij,...i->j..."


class TestEinsumWithBroadcast3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "k...,...jk->...k"


class TestEinsumWithBroadcast4(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "a...d,...cb->...abcd"


class TestEinsumWithBroadcast5(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 2, 2, 10), (10, 3, 2, 2)]
        self.types = [np.float64, np.float64]
        self.equation = "...a,a...->..."


class TestEinsumWithBroadcast6(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(100), (100)]
        self.types = [np.float64, np.float64]
        self.equation = "i,i->"


if __name__ == "__main__":
    unittest.main()
