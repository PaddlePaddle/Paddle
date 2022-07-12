# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid.core as core

paddle.enable_static()


class TestBucketizeAPI(unittest.TestCase):

    def init_test_case(self):
        self.x = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype("float32")
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype("float32")

    def setUp(self):
        self.init_test_case()
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('InputTensor',
                                       shape=self.x.shape,
                                       dtype="float64")
                sorted_sequence = paddle.static.data(
                    'SortedSequence',
                    shape=self.sorted_sequence.shape,
                    dtype="float64")
                out = paddle.bucketize(x, sorted_sequence)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'InputTensor': self.x,
                    'SortedSequence': self.sorted_sequence,
                },
                              fetch_list=out)
                out_ref = np.searchsorted(self.sorted_sequence, self.x)
                self.assertTrue(np.allclose(out_ref, res))

        for place in self.place:
            run(place)

    def test_dygraph_api(self):

        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            out = paddle.bucketize(x, sorted_sequence, right=True)
            out_ref = np.searchsorted(self.sorted_sequence,
                                      self.x,
                                      side='right')
            self.assertEqual(np.allclose(out_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_out_int32(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        sorted_sequence = paddle.to_tensor(self.sorted_sequence)
        out = paddle.bucketize(x, sorted_sequence, out_int32=True)
        self.assertTrue(out.type, 'int32')

    def test_right(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        sorted_sequence = paddle.to_tensor(self.sorted_sequence)
        out = paddle.bucketize(x, sorted_sequence, right=True)
        out_ref = np.searchsorted(self.sorted_sequence, self.x, side='right')
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)


class TestBucketizeError(unittest.TestCase):

    def init_test_case(self):
        self.x = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype("float32")
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype("float32")

    def setUp(self):
        self.init_test_case()
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_error_api(self):
        paddle.enable_static()

        def test_bucketize_searchsorted_dims_error():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('InputTensor',
                                       shape=[2, 5],
                                       dtype="float64")
                sorted_sequence = paddle.static.data('SortedSequence',
                                                     shape=[2, 2, 3],
                                                     dtype="float64")
                out = paddle.bucketize(x, sorted_sequence)

        self.assertRaises(AssertionError,
                          test_bucketize_searchsorted_dims_error)

        def test_sortedsequence_x_type_error():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('InputTensor',
                                       shape=[2, 5],
                                       dtype="int16")
                sorted_sequence = paddle.static.data('SortedSequence',
                                                     shape=[2, 3],
                                                     dtype="int16")
                out = paddle.bucketize(x, sorted_sequence)

        self.assertRaises(TypeError, test_sortedsequence_x_type_error)

        def test_tensor_type():
            paddle.disable_static()
            x = np.ones((3, 3))
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            out = paddle.bucketize(x, sorted_sequence)

        self.assertRaises(TypeError, test_tensor_type)
