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

from __future__ import print_function
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
                out = paddle.bucketize(x, sorted_sequence)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'SortedSequence': self.sorted_sequence,
                    'x': self.x
                },
                              fetch_list=out)
                out_ref = np.searchsorted(self.sorted_sequence, self.x)
                self.assertTrue(np.allclose(out_ref, res))

        for place in self.place:
            run(place)

    def test_api_dygraph(self):

        def run(place):
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
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


if __name__ == "__main__":
    unittest.main()
