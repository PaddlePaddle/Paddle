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

import unittest

import numpy as np
from op_test import get_places

import paddle

np.random.seed(10)


class TestBucketizeAPI(unittest.TestCase):
    # test paddle.tensor.math.nanmean

    def setUp(self):
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype("float64")
        self.x = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype("float64")
        self.place = get_places()

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence',
                    shape=self.sorted_sequence.shape,
                    dtype="float64",
                )
                x = paddle.static.data('x', shape=self.x.shape, dtype="float64")
                out1 = paddle.bucketize(x, sorted_sequence)
                out2 = paddle.bucketize(x, sorted_sequence, right=True)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={'SortedSequence': self.sorted_sequence, 'x': self.x},
                    fetch_list=[out1, out2],
                )
            out_ref = np.searchsorted(self.sorted_sequence, self.x)
            out_ref1 = np.searchsorted(
                self.sorted_sequence, self.x, side='right'
            )
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)
            np.testing.assert_allclose(out_ref1, res[1], rtol=1e-05)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
            out1 = paddle.bucketize(x, sorted_sequence)
            out2 = paddle.bucketize(x, sorted_sequence, right=True)
            out_ref1 = np.searchsorted(self.sorted_sequence, self.x)
            out_ref2 = np.searchsorted(
                self.sorted_sequence, self.x, side='right'
            )
            np.testing.assert_allclose(out_ref1, out1.numpy(), rtol=1e-05)
            np.testing.assert_allclose(out_ref2, out2.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_out_int32(self):
        paddle.disable_static()
        sorted_sequence = paddle.to_tensor(self.sorted_sequence)
        x = paddle.to_tensor(self.x)
        out = paddle.bucketize(x, sorted_sequence, out_int32=True)
        self.assertTrue(out.type, 'int32')

    def test_bucketize_dims_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            sorted_sequence = paddle.static.data(
                'SortedSequence', shape=[2, 2], dtype="float64"
            )
            x = paddle.static.data('x', shape=[2, 5], dtype="float64")
            self.assertRaises(ValueError, paddle.bucketize, x, sorted_sequence)

    def test_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            self.assertRaises(
                ValueError, paddle.bucketize, self.x, sorted_sequence
            )

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
            self.assertRaises(
                ValueError, paddle.bucketize, None, sorted_sequence
            )
            self.assertRaises(AttributeError, paddle.bucketize, x, None)


class TestBucketizeAPI_Extended(unittest.TestCase):
    def setUp(self):
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype("float64")
        self.x2d = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype("float64")
        self.x1d = np.array([0, 8, 4, 16]).astype("float64")
        self.sorted_dup = np.array([1, 2, 2, 2, 3]).astype("float64")
        self.x_dup = np.array([2, 2, 1, 3]).astype("float64")
        self.place = get_places()

    def test_dygraph_out_and_out_int32_and_name(self):
        # Dynamic diagram: Testing the out parameter (inplace write) and out_int32
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                x = paddle.to_tensor(self.x2d)

                res32 = paddle.bucketize(
                    x, seq, out_int32=True, name="test_name"
                )
                self.assertEqual(res32.dtype, paddle.int32)
                ref32 = np.searchsorted(self.sorted_sequence, self.x2d)
                np.testing.assert_allclose(
                    ref32, res32.numpy().astype("int64"), rtol=1e-05
                )

                # out parameter: supply existing tensor, should be written and returned
                out_tensor = paddle.empty(shape=self.x2d.shape, dtype="int64")
                ret = paddle.bucketize(x, seq, out=out_tensor)
                ref = np.searchsorted(self.sorted_sequence, self.x2d)
                np.testing.assert_allclose(ref, out_tensor.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_out_int32_and_right(self):
        # Static image: Testing out_int32 and right=True/False
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                seq = paddle.static.data(
                    name="seq",
                    shape=self.sorted_sequence.shape,
                    dtype="float64",
                )
                x = paddle.static.data(
                    name="x", shape=self.x2d.shape, dtype="float64"
                )

                out_left = paddle.bucketize(
                    x, seq, right=False, out_int32=False
                )
                out_right = paddle.bucketize(x, seq, right=True, out_int32=True)

                exe = paddle.static.Executor(place)
                res_left, res_right = exe.run(
                    feed={"seq": self.sorted_sequence, "x": self.x2d},
                    fetch_list=[out_left, out_right],
                )
                ref_left = np.searchsorted(
                    self.sorted_sequence, self.x2d, side="left"
                )
                ref_right = np.searchsorted(
                    self.sorted_sequence, self.x2d, side="right"
                )
                np.testing.assert_allclose(ref_left, res_left, rtol=1e-05)
                # out_int32 True -> numpy result must be cast-compatible to int32
                self.assertEqual(res_right.dtype, np.int32)
                np.testing.assert_allclose(
                    ref_right, res_right.astype("int64"), rtol=1e-05
                )
        paddle.disable_static()

    def test_dygraph_1d_input(self):
        # Dynamic image: 1D x test
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                x = paddle.to_tensor(self.x1d)

                out = paddle.bucketize(x, seq)
                ref = np.searchsorted(self.sorted_sequence, self.x1d)
                np.testing.assert_allclose(ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_dup_elements_side_behavior(self):
        # Left/right difference when testing duplicate elements
        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_dup)
                x = paddle.to_tensor(self.x_dup)

                out_left = paddle.bucketize(x, seq, right=False)
                out_right = paddle.bucketize(x, seq, right=True)

                ref_left = np.searchsorted(
                    self.sorted_dup, self.x_dup, side="left"
                )
                ref_right = np.searchsorted(
                    self.sorted_dup, self.x_dup, side="right"
                )

                np.testing.assert_allclose(
                    ref_left, out_left.numpy(), rtol=1e-05
                )
                np.testing.assert_allclose(
                    ref_right, out_right.numpy(), rtol=1e-05
                )
        paddle.enable_static()

    def test_static_and_dygraph_sort_of_api_stability(self):
        # Simple coverage: Both static and dynamic calls can succeed (without checking for duplicate results)
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                seq = paddle.static.data(
                    name="seq",
                    shape=self.sorted_sequence.shape,
                    dtype="float64",
                )
                x = paddle.static.data(
                    name="x", shape=self.x2d.shape, dtype="float64"
                )
                _ = paddle.bucketize(
                    x, seq, out_int32=False, right=False, name="static_case"
                )
                exe = paddle.static.Executor(place)
                exe.run(
                    feed={"seq": self.sorted_sequence, "x": self.x2d},
                    fetch_list=[],
                )
        paddle.disable_static()

        paddle.disable_static()
        for place in self.place:
            with paddle.base.dygraph.guard():
                seq = paddle.to_tensor(self.sorted_sequence)
                x = paddle.to_tensor(self.x2d)
                _ = paddle.bucketize(
                    x, seq, out_int32=False, right=False, name="dy_case"
                )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
