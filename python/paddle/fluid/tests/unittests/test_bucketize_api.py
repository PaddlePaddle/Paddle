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

from re import X

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

np.random.seed(10)


class TestBucketizeAPI(unittest.TestCase):
    # test paddle.tensor.math.nanmean

    def setUp(self):
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype("float64")
        self.x = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype("float64")
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data(
                    'SortedSequence',
                    shape=self.sorted_sequence.shape,
                    dtype="float64")
                x = paddle.static.data('x', shape=self.x.shape, dtype="float64")
                out1 = paddle.bucketize(x, sorted_sequence)
                out2 = paddle.bucketize(x, sorted_sequence, right=True)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'SortedSequence': self.sorted_sequence,
                    'x': self.x
                },
                              fetch_list=[out1, out2])
            out_ref = np.searchsorted(self.sorted_sequence, self.x)
            out_ref1 = np.searchsorted(self.sorted_sequence,
                                       self.x,
                                       side='right')
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
            out_ref2 = np.searchsorted(self.sorted_sequence,
                                       self.x,
                                       side='right')
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
            sorted_sequence = paddle.static.data('SortedSequence',
                                                 shape=[2, 2],
                                                 dtype="float64")
            x = paddle.static.data('x', shape=[2, 5], dtype="float64")
            self.assertRaises(ValueError, paddle.bucketize, x, sorted_sequence)

    def test_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            self.assertRaises(ValueError, paddle.bucketize, self.x,
                              sorted_sequence)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
            self.assertRaises(ValueError, paddle.bucketize, None,
                              sorted_sequence)
            self.assertRaises(AttributeError, paddle.bucketize, x, None)


if __name__ == "__main__":
    unittest.main()
