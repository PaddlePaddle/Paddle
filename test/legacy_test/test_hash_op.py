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

import unittest

import numpy as np
from op_test import OpTest


class TestHashOp(OpTest):
    def setUp(self):
        self.op_type = "hash"
        self.init_test_case()
        self.inputs = {'X': (self.in_seq, self.lod)}
        self.attrs = {'num_hash': 2, 'mod_by': 10000}
        self.outputs = {'Out': (self.out_seq, self.lod)}

    def init_test_case(self):
        np.random.seed(1)
        self.in_seq = np.random.randint(0, 10, (8, 1)).astype("int32")
        self.lod = [[2, 6]]
        self.out_seq = [
            [[3481], [7475]],
            [[1719], [5986]],
            [[8473], [694]],
            [[3481], [7475]],
            [[4372], [9456]],
            [[4372], [9456]],
            [[6897], [3218]],
            [[9038], [7951]],
        ]
        self.out_seq = np.array(self.out_seq)

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestHashNotLoDOp(TestHashOp):
    def setUp(self):
        self.op_type = "hash"
        self.init_test_case()
        self.inputs = {'X': self.in_seq}
        self.attrs = {'num_hash': 2, 'mod_by': 10000}
        self.outputs = {'Out': self.out_seq}

    def init_test_case(self):
        np.random.seed(1)
        self.in_seq = np.random.randint(0, 10, (8, 1)).astype("int32")
        self.out_seq = [
            [[3481], [7475]],
            [[1719], [5986]],
            [[8473], [694]],
            [[3481], [7475]],
            [[4372], [9456]],
            [[4372], [9456]],
            [[6897], [3218]],
            [[9038], [7951]],
        ]
        self.out_seq = np.array(self.out_seq)

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestHashOp2(TestHashOp):
    """
    Case:
    int64 type input
    """

    def setUp(self):
        self.op_type = "hash"
        self.init_test_case()
        self.inputs = {'X': self.in_seq}
        self.attrs = {'num_hash': 2, 'mod_by': 10000}
        self.outputs = {'Out': self.out_seq}

    def init_test_case(self):
        self.in_seq = np.array([1, 2**32 + 1]).reshape((2, 1)).astype("int64")
        self.out_seq = np.array([1269, 9609, 3868, 7268]).reshape((2, 2, 1))

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestHashOp3(TestHashOp):
    """
    Case:
    int64 type input
    int64 type mod_by attr
    """

    def setUp(self):
        self.op_type = "hash"
        self.init_test_case()
        self.inputs = {'X': self.in_seq}
        self.attrs = {'num_hash': 2, 'mod_by': 2**32}
        self.outputs = {'Out': self.out_seq}

    def init_test_case(self):
        self.in_seq = np.array([10, 5]).reshape((2, 1)).astype("int64")
        self.out_seq = np.array(
            [1204014882, 393011615, 3586283837, 2814821595]
        ).reshape((2, 2, 1))

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == "__main__":
    unittest.main()
