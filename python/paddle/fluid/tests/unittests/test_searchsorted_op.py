# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()
from op_test import OpTest


class TestSearchSorted(OpTest):
    def setUp(self):

        self.op_type = "searchsorted"
        self.init_dtype()
        self.init_test_case()

        self.inputs = {
            'SortedSequence': self.sorted_sequence,
            'Values': self.values
        }
        self.attrs = {"out_int32": False, "right": False}
        self.side = "right" if self.attrs["right"] else "left"
        self.outputs = {
            'Out': np.searchsorted(
                self.sorted_sequence, self.values, side=self.side)
        }

    def test_check_output(self):
        self.check_output()

    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9])
        self.values = np.array([[3, 6, 9], [3, 6, 9]])
        self.side = "left"

    def init_dtype(self):
        self.dtype = np.float64


class TestSearchSorted_float32(TestSearchSorted):
    def init_dtype(self):
        self.dtype = np.float32


class TestSearchSorted_int32(TestSearchSorted):
    def init_dtype(self):
        self.dtype = np.int32


class TestSearchSorted_int64(TestSearchSorted):
    def init_dtype(self):
        self.dtype = np.int64


class TestSearchSortedAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.int64

    def setUp(self):
        self.init_dtype()
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype(self.dtype)
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype(self.dtype)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence',
                    shape=self.sorted_sequence.shape,
                    dtype=self.dtype)
                values = paddle.static.data(
                    'Values', shape=self.values.shape, dtype=self.dtype)
                out = paddle.searchsorted(sorted_sequence, values)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'SortedSequence': self.sorted_sequence,
                    'Values': self.values
                },
                              fetch_list=out)
            out_ref = np.searchsorted(self.sorted_sequence, self.values)
            for r in res:
                self.assertEqual(np.allclose(out_ref, r), True)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):

            paddle.disable_static(place)
            SortedSequence = paddle.to_tensor(self.sorted_sequence)
            Values = paddle.to_tensor(self.values)
            out = paddle.searchsorted(SortedSequence, Values)
            out_ref = np.searchsorted(self.sorted_sequence, self.values)
            self.assertEqual(np.allclose(out_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
