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
import paddle.fluid.core as core

paddle.enable_static()


class TestSearchSorted(OpTest):
    def setUp(self):
        self.python_api = paddle.searchsorted
        self.op_type = "searchsorted"
        self.init_test_case()

        self.inputs = {
            'SortedSequence': self.sorted_sequence,
            'Values': self.values,
        }
        self.attrs = {"out_int32": False, "right": False}
        self.attrs["right"] = True if self.side == 'right' else False
        self.outputs = {
            'Out': np.searchsorted(
                self.sorted_sequence, self.values, side=self.side
            )
        }

    def test_check_output(self):
        self.check_output(check_eager=True)

    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("float32")
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype("float32")
        self.side = "left"


class TestSearchSortedOp1(TestSearchSorted):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("int32")
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype("int32")
        self.side = "right"


class TestSearchSortedOp2(TestSearchSorted):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("int64")
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype("int64")
        self.side = "left"


class TestSearchSortedOp3(TestSearchSorted):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("float64")
        self.values = np.array([[np.nan, np.nan, np.nan], [3, 6, 9]]).astype(
            "float64"
        )
        self.side = "left"


class TestSearchSortedOp4(TestSearchSorted):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("float64")
        self.values = np.array([[np.inf, np.inf, np.inf], [3, 6, 9]]).astype(
            "float64"
        )
        self.side = "right"


class TestSearchSortedOp5(TestSearchSorted):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("float64")
        self.values = np.array(
            [[np.inf, np.inf, np.inf], [np.nan, np.nan, np.nan]]
        ).astype("float64")
        self.side = "right"


class TestSearchSortedAPI(unittest.TestCase):
    def init_test_case(self):
        self.sorted_sequence = np.array([2, 4, 6, 8, 10]).astype("float64")
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype("float64")

    def setUp(self):
        self.init_test_case()
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
                    dtype="float64",
                )
                values = paddle.static.data(
                    'Values', shape=self.values.shape, dtype="float64"
                )
                out = paddle.searchsorted(sorted_sequence, values)
                exe = paddle.static.Executor(place)
                (res,) = exe.run(
                    feed={
                        'SortedSequence': self.sorted_sequence,
                        'Values': self.values,
                    },
                    fetch_list=out,
                )
            out_ref = np.searchsorted(self.sorted_sequence, self.values)
            np.testing.assert_allclose(out_ref, res, rtol=1e-05)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):

            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            values = paddle.to_tensor(self.values)
            out = paddle.searchsorted(sorted_sequence, values, right=True)
            out_ref = np.searchsorted(
                self.sorted_sequence, self.values, side='right'
            )
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_out_int32(self):
        paddle.disable_static()
        sorted_sequence = paddle.to_tensor(self.sorted_sequence)
        values = paddle.to_tensor(self.values)
        out = paddle.searchsorted(sorted_sequence, values, out_int32=True)
        self.assertTrue(out.type, 'int32')


class TestSearchSortedError(unittest.TestCase):
    def test_error_api(self):
        paddle.enable_static()

        def test_searchsorted_dims_matched_before_lastdim_error1():
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence', shape=[2, 2, 3], dtype="float64"
                )
                values = paddle.static.data(
                    'Values', shape=[2, 5], dtype="float64"
                )
                out = paddle.searchsorted(sorted_sequence, values)

        self.assertRaises(
            RuntimeError, test_searchsorted_dims_matched_before_lastdim_error1
        )

        def test_searchsorted_dims_matched_before_lastdim_error2():
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence', shape=[2, 2, 3], dtype="float64"
                )
                values = paddle.static.data(
                    'Values', shape=[2, 3, 5], dtype="float64"
                )
                out = paddle.searchsorted(sorted_sequence, values)

        self.assertRaises(
            RuntimeError, test_searchsorted_dims_matched_before_lastdim_error2
        )

        def test_searchsorted_sortedsequence_size_error():
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence', shape=[2, 2, pow(2, 34)], dtype="float64"
                )
                values = paddle.static.data(
                    'Values', shape=[2, 2, 5], dtype="float64"
                )
                out = paddle.searchsorted(
                    sorted_sequence, values, out_int32=True
                )

        self.assertRaises(
            RuntimeError, test_searchsorted_sortedsequence_size_error
        )

        def test_sortedsequence_values_type_error():
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence', shape=[2, 3], dtype="int16"
                )
                values = paddle.static.data(
                    'Values', shape=[2, 5], dtype="int16"
                )
                out = paddle.searchsorted(sorted_sequence, values)

        self.assertRaises(TypeError, test_sortedsequence_values_type_error)


if __name__ == '__main__':
    unittest.main()
