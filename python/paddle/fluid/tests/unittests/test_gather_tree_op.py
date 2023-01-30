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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.fluid.framework import Program, program_guard


class TestGatherTreeOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import program_guard, Program


class TestGatherTreeOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "gather_tree"
        self.python_api = paddle.nn.functional.gather_tree
        max_length, batch_size, beam_size = 5, 2, 2
<<<<<<< HEAD
        ids = np.random.randint(
            0, high=10, size=(max_length, batch_size, beam_size)
        )
        parents = np.random.randint(
            0, high=beam_size, size=(max_length, batch_size, beam_size)
        )
=======
        ids = np.random.randint(0,
                                high=10,
                                size=(max_length, batch_size, beam_size))
        parents = np.random.randint(0,
                                    high=beam_size,
                                    size=(max_length, batch_size, beam_size))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {"Ids": ids, "Parents": parents}
        self.outputs = {'Out': self.backtrace(ids, parents)}

    def test_check_output(self):
        self.check_output(check_eager=True)

    @staticmethod
    def backtrace(ids, parents):
        out = np.zeros_like(ids)
        (max_length, batch_size, beam_size) = ids.shape
        for batch in range(batch_size):
            for beam in range(beam_size):
<<<<<<< HEAD
                out[max_length - 1, batch, beam] = ids[
                    max_length - 1, batch, beam
                ]
=======
                out[max_length - 1, batch, beam] = ids[max_length - 1, batch,
                                                       beam]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                parent = parents[max_length - 1, batch, beam]
                for step in range(max_length - 2, -1, -1):
                    out[step, batch, beam] = ids[step, batch, parent]
                    parent = parents[step, batch, parent]
        return out


class TestGatherTreeOpAPI(unittest.TestCase):
<<<<<<< HEAD
    def test_case(self):
        paddle.enable_static()
        ids = paddle.static.data(name='ids', shape=[5, 2, 2], dtype='int64')
        parents = paddle.static.data(
            name='parents',
            shape=[5, 2, 2],
            dtype='int64',
        )
        final_sequences = paddle.nn.functional.gather_tree(ids, parents)
        paddle.disable_static()

    def test_case2(self):
        ids = paddle.to_tensor(
            [[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]
        )
        parents = paddle.to_tensor(
            [[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]
        )
=======

    def test_case(self):
        paddle.enable_static()
        ids = fluid.layers.data(name='ids',
                                shape=[5, 2, 2],
                                dtype='int64',
                                append_batch_size=False)
        parents = fluid.layers.data(name='parents',
                                    shape=[5, 2, 2],
                                    dtype='int64',
                                    append_batch_size=False)
        final_sequences = fluid.layers.gather_tree(ids, parents)
        paddle.disable_static()

    def test_case2(self):
        ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]],
                                [[0, 1], [9, 0]]])
        parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]],
                                    [[0, 0], [0, 1]]])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        final_sequences = paddle.nn.functional.gather_tree(ids, parents)


class TestGatherTreeOpError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            ids = paddle.static.data(name='ids', shape=[5, 2, 2], dtype='int64')
            parents = paddle.static.data(
                name='parents', shape=[5, 2, 2], dtype='int64'
            )
=======

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            ids = fluid.layers.data(name='ids',
                                    shape=[5, 2, 2],
                                    dtype='int64',
                                    append_batch_size=False)
            parents = fluid.layers.data(name='parents',
                                        shape=[5, 2, 2],
                                        dtype='int64',
                                        append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            def test_Variable_ids():
                # the input type must be Variable
                np_ids = np.random.random((5, 2, 2), dtype='int64')
<<<<<<< HEAD
                paddle.nn.functional.gather_tree(np_ids, parents)
=======
                fluid.layers.gather_tree(np_ids, parents)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_Variable_ids)

            def test_Variable_parents():
                # the input type must be Variable
                np_parents = np.random.random((5, 2, 2), dtype='int64')
<<<<<<< HEAD
                paddle.nn.functional.gather_tree(ids, np_parents)
=======
                fluid.layers.gather_tree(ids, np_parents)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_Variable_parents)

            def test_type_ids():
                # dtype must be int32 or int64
<<<<<<< HEAD
                bad_ids = paddle.static.data(
                    name='bad_ids', shape=[5, 2, 2], dtype='float32'
                )
                paddle.nn.functional.gather_tree(bad_ids, parents)
=======
                bad_ids = fluid.layers.data(name='bad_ids',
                                            shape=[5, 2, 2],
                                            dtype='float32',
                                            append_batch_size=False)
                fluid.layers.gather_tree(bad_ids, parents)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_type_ids)

            def test_type_parents():
                # dtype must be int32 or int64
<<<<<<< HEAD
                bad_parents = paddle.static.data(
                    name='bad_parents', shape=[5, 2, 2], dtype='float32'
                )
                paddle.nn.functional.gather_tree(ids, bad_parents)
=======
                bad_parents = fluid.layers.data(name='bad_parents',
                                                shape=[5, 2, 2],
                                                dtype='float32',
                                                append_batch_size=False)
                fluid.layers.gather_tree(ids, bad_parents)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_type_parents)

            def test_ids_ndim():
<<<<<<< HEAD
                bad_ids = paddle.static.data(
                    name='bad_test_ids', shape=[5, 2], dtype='int64'
                )
=======
                bad_ids = fluid.layers.data(name='bad_test_ids',
                                            shape=[5, 2],
                                            dtype='int64',
                                            append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.nn.functional.gather_tree(bad_ids, parents)

            self.assertRaises(ValueError, test_ids_ndim)

            def test_parents_ndim():
<<<<<<< HEAD
                bad_parents = paddle.static.data(
                    name='bad_test_parents', shape=[5, 2], dtype='int64'
                )
=======
                bad_parents = fluid.layers.data(name='bad_test_parents',
                                                shape=[5, 2],
                                                dtype='int64',
                                                append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.nn.functional.gather_tree(ids, bad_parents)

            self.assertRaises(ValueError, test_parents_ndim)

        paddle.disable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
