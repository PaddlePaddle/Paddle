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
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid.framework import program_guard, Program


class TestGatherTreeOp(OpTest):
    def setUp(self):
        self.op_type = "gather_tree"
        max_length, batch_size, beam_size = 5, 2, 2
        ids = np.random.randint(
            0, high=10, size=(max_length, batch_size, beam_size))
        parents = np.random.randint(
            0, high=beam_size, size=(max_length, batch_size, beam_size))
        self.inputs = {"Ids": ids, "Parents": parents}
        self.outputs = {'Out': self.backtrace(ids, parents)}

    def test_check_output(self):
        self.check_output()

    @staticmethod
    def backtrace(ids, parents):
        out = np.zeros_like(ids)
        (max_length, batch_size, beam_size) = ids.shape
        for batch in range(batch_size):
            for beam in range(beam_size):
                out[max_length - 1, batch, beam] = ids[max_length - 1, batch,
                                                       beam]
                parent = parents[max_length - 1, batch, beam]
                for step in range(max_length - 2, -1, -1):
                    out[step, batch, beam] = ids[step, batch, parent]
                    parent = parents[step, batch, parent]
        return out


class TestGatherTreeOpAPI(unittest.TestCase):
    def test_case(self):
        ids = fluid.layers.data(
            name='ids', shape=[5, 2, 2], dtype='int64', append_batch_size=False)
        parents = fluid.layers.data(
            name='parents',
            shape=[5, 2, 2],
            dtype='int64',
            append_batch_size=False)
        final_sequences = fluid.layers.gather_tree(ids, parents)


class TestGatherTreeOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            ids = fluid.layers.data(
                name='ids',
                shape=[5, 2, 2],
                dtype='int64',
                append_batch_size=False)
            parents = fluid.layers.data(
                name='parents',
                shape=[5, 2, 2],
                dtype='int64',
                append_batch_size=False)

            def test_Variable_ids():
                # the input type must be Variable
                np_ids = np.random.random((5, 2, 2), dtype='int64')
                fluid.layers.gather_tree(np_ids, parents)

            self.assertRaises(TypeError, test_Variable_ids)

            def test_Variable_parents():
                # the input type must be Variable
                np_parents = np.random.random((5, 2, 2), dtype='int64')
                fluid.layers.gather_tree(ids, np_parents)

            self.assertRaises(TypeError, test_Variable_parents)

            def test_type_ids():
                # dtype must be int32 or int64
                bad_ids = fluid.layers.data(
                    name='bad_ids',
                    shape=[5, 2, 2],
                    dtype='float32',
                    append_batch_size=False)
                fluid.layers.gather_tree(bad_ids, parents)

            self.assertRaises(TypeError, test_type_ids)

            def test_type_parents():
                # dtype must be int32 or int64
                bad_parents = fluid.layers.data(
                    name='bad_parents',
                    shape=[5, 2, 2],
                    dtype='float32',
                    append_batch_size=False)
                fluid.layers.gather_tree(ids, bad_parents)

            self.assertRaises(TypeError, test_type_parents)


if __name__ == "__main__":
    unittest.main()
