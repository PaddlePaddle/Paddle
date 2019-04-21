# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestSequenceConcat(OpTest):
    def setLoD(self):
        self.lod1 = [7, 3]
        self.lod2 = [12, 8]
        self.out_lod = [19, 11]

    def setUp(self):
        self.test_gc = True
        x1 = np.random.random(size=(10, 80))
        x2 = np.random.random(size=(20, 80))
        self.setLoD()

        out = np.concatenate((x1[0:self.lod1[0]], x2[0:self.lod2[0]],
                              x1[self.lod1[0]:], x2[self.lod2[0]:]))

        self.op_type = "sequence_concat"
        self.inputs = {
            'X': [("x1", (x1, [self.lod1])), ("x2", (x2, [self.lod2]))]
        }
        self.outputs = {"Out": (out, [self.out_lod])}

    def test_output(self):
        self.check_output(1e-3)

    def test_dx(self):
        self.check_grad(inputs_to_check=['x1', 'x2'], output_names="Out")


class TestSequenceConcatCase2(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [10, 0]
        self.lod2 = [12, 8]
        self.out_lod = [22, 8]


class TestSequenceConcatCase3(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [10, 0]
        self.lod2 = [20, 0]
        self.out_lod = [30, 0]


class TestSequenceConcatCase4(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [0, 10]
        self.lod2 = [0, 20]
        self.out_lod = [0, 30]


class TestSequenceConcatCase5(TestSequenceConcat):
    def setLoD(self):
        self.lod1 = [0, 10]
        self.lod2 = [20, 0]
        self.out_lod = [20, 10]


if __name__ == '__main__':
    unittest.main()
