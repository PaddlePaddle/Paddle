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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    get_places,
    is_custom_device,
)

import paddle
from paddle.base import core

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
        self.check_output(check_pir=True)

    def init_shape(self):
        self.shape = None

    def init_test_case(self):
        self.init_shape()
        if self.shape is None:
            self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype("float32")
        else:
            self.sorted_sequence = np.random.randn(*self.shape).astype(
                "float32"
            )
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


class TestSearchSorted_ZeroSize(TestSearchSorted):
    def init_shape(self):
        self.shape = (0,)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_float16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the float16",
)
class TestSearchSortedFP16OP(TestSearchSorted):
    def setUp(self):
        self.python_api = paddle.searchsorted
        self.op_type = "searchsorted"
        self.dtype = np.float16
        self.init_test_case()

        self.inputs = {
            'SortedSequence': self.sorted_sequence.astype(self.dtype),
            'Values': self.values.astype(self.dtype),
        }
        self.attrs = {"out_int32": False, "right": False}
        self.attrs["right"] = True if self.side == 'right' else False
        self.outputs = {
            'Out': np.searchsorted(
                self.sorted_sequence, self.values, side=self.side
            )
        }

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(place, check_pir=True)

    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9])
        self.values = np.array([[3, 6, 9], [3, 6, 9]])
        self.side = "left"


class TestSearchSortedFP16OP_2(TestSearchSortedFP16OP):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9])
        self.values = np.array([[3, 6, 9], [3, 6, 9]])
        self.side = "right"


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestSearchSortedBF16(TestSearchSorted):
    def setUp(self):
        self.python_api = paddle.searchsorted
        self.public_python_api = paddle.searchsorted
        self.op_type = "searchsorted"
        self.python_out_sig = ["Out"]
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.init_test_case()

        self.inputs = {
            'SortedSequence': convert_float_to_uint16(self.sorted_sequence),
            'Values': convert_float_to_uint16(self.values),
        }
        self.attrs = {"out_int32": False, "right": False}
        self.attrs["right"] = True if self.side == 'right' else False
        self.outputs = {
            'Out': np.searchsorted(
                self.sorted_sequence, self.values, side=self.side
            )
        }

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(place, check_pir=True)

    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype(self.np_dtype)
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype(self.np_dtype)
        self.side = "left"


class TestSearchSortedBF16_2(TestSearchSortedBF16):
    def init_test_case(self):
        self.sorted_sequence = np.array([1, 3, 5, 7, 9]).astype(self.np_dtype)
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype(self.np_dtype)
        self.side = "right"


class TestSearchSortedAPI(unittest.TestCase):
    def init_test_case(self):
        self.sorted_sequence = np.array([2, 4, 6, 8, 10]).astype("float64")
        self.values = np.array([[3, 6, 9], [3, 6, 9]]).astype("float64")

    def setUp(self):
        self.init_test_case()
        self.place = get_places()

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
            with paddle.base.dygraph.guard():
                sorted_sequence = paddle.to_tensor(self.sorted_sequence)
                values = paddle.to_tensor(self.values)
                out = paddle.searchsorted(sorted_sequence, values, right=True)
                out_ref = np.searchsorted(
                    self.sorted_sequence, self.values, side='right'
                )
                np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

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

    def test_check_type_error(self):
        paddle.enable_static()

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


class TestSearchSortedAPI_Extended(unittest.TestCase):
    def init_test_case(self):
        self.sorted_sequence = np.array([2, 4, 6, 8, 10]).astype("float64")
        self.values_2d = np.array([[3, 6, 9], [3, 6, 9]]).astype("float64")
        self.values_1d = np.array([3, 6, 9]).astype("float64")
        self.unsorted_seq = np.array([6, 2, 10, 4, 8]).astype("float64")
        # sorter such that unsorted_seq[sorter] is sorted
        self.sorter = np.argsort(self.unsorted_seq).astype("int64")

    def setUp(self):
        self.init_test_case()
        self.place = get_places()

    def test_dygraph_side_and_right_priority_and_out_int32(self):
        # Test: side takes precedence over right, out_int32 controls the returned dtype
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                vals = paddle.to_tensor(self.values_2d)
                # Mixed parameter passing: right=False, side='right' -> side takes precedence, should be interpreted as right=True
                out = paddle.searchsorted(
                    seq, vals, right=False, side="right", out_int32=True
                )
                ref = np.searchsorted(
                    self.sorted_sequence, self.values_2d, side="right"
                )
                self.assertEqual(out.dtype, paddle.int32)
                np.testing.assert_allclose(
                    ref, out.numpy().astype("int64"), rtol=1e-05
                )

    def test_dygraph_out_parameter_and_return_is_out(self):
        # Test out parameter: Pass in an existing tensor, write the function on it, and return the same Tensor
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                vals = paddle.to_tensor(self.values_2d)
                out_tensor = paddle.empty(
                    shape=self.values_2d.shape, dtype="int64"
                )
                ret = paddle.searchsorted(seq, vals, out=out_tensor)
                ref = np.searchsorted(self.sorted_sequence, self.values_2d)
                np.testing.assert_allclose(ref, out_tensor.numpy(), rtol=1e-05)

    def test_dygraph_sorter_behavior(self):
        # Test sorter parameter: When the sequence is unsorted but a sorter is given, the behavior is consistent with numpy
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.unsorted_seq)
                vals = paddle.to_tensor(self.values_1d)
                sorter_t = paddle.to_tensor(self.sorter)
                out = paddle.searchsorted(seq, vals, sorter=sorter_t)
                ref = np.searchsorted(
                    self.unsorted_seq, self.values_1d, sorter=self.sorter
                )
                np.testing.assert_allclose(ref, out.numpy(), rtol=1e-05)

    def test_static_side_and_sorter(self):
        # Test side parameters and sorter parameters under static images (aligned with numpy)
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                seq = paddle.static.data(
                    name="seq", shape=self.unsorted_seq.shape, dtype="float64"
                )
                vals = paddle.static.data(
                    name="vals", shape=self.values_1d.shape, dtype="float64"
                )
                sorter = paddle.static.data(
                    name="sorter", shape=self.sorter.shape, dtype="int64"
                )

                out_left = paddle.searchsorted(
                    seq, vals, side="left", sorter=sorter
                )
                out_right = paddle.searchsorted(
                    seq, vals, side="right", sorter=sorter
                )

                exe = paddle.static.Executor(place)
                (res_left, res_right) = exe.run(
                    feed={
                        "seq": self.unsorted_seq,
                        "vals": self.values_1d,
                        "sorter": self.sorter,
                    },
                    fetch_list=[out_left, out_right],
                )
                ref_left = np.searchsorted(
                    self.unsorted_seq,
                    self.values_1d,
                    side="left",
                    sorter=self.sorter,
                )
                ref_right = np.searchsorted(
                    self.unsorted_seq,
                    self.values_1d,
                    side="right",
                    sorter=self.sorter,
                )
                np.testing.assert_allclose(ref_left, res_left, rtol=1e-05)
                np.testing.assert_allclose(ref_right, res_right, rtol=1e-05)
        paddle.disable_static()

    def test_dygraph_1d_values_and_name_param(self):
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                vals = paddle.to_tensor(self.values_1d)
                out = paddle.searchsorted(seq, vals, name="my_search")
                ref = np.searchsorted(self.sorted_sequence, self.values_1d)
                np.testing.assert_allclose(ref, out.numpy(), rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
