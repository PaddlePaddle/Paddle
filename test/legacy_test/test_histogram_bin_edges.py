# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestHistogramBinEdgesAPI(unittest.TestCase):
    """Test histogram_bin_edges api."""

    def setUp(self):
        self.input_np = np.random.uniform(-5, 5, [2, 3]).astype(np.float32)
        self.bins = 4
        self.range = (0.0, 3.0)
        self.place = [paddle.CPUPlace()]

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                inputs = paddle.static.data(
                    name='input', dtype='float32', shape=[2, 3]
                )
                out = paddle.histogram_bin_edges(inputs, self.bins, self.range)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={'input': self.input_np},
                    fetch_list=[out],
                )
            out_ref = np.histogram_bin_edges(
                self.input_np, self.bins, self.range
            )
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            inputs = paddle.to_tensor(self.input_np)
            out1 = paddle.histogram_bin_edges(inputs, bins=4, range=(0, 3))
            out_ref1 = np.histogram_bin_edges(
                self.input_np, bins=4, range=(0, 3)
            )
            np.testing.assert_allclose(out_ref1, out1.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_errors(self):
        input = paddle.to_tensor(self.input_np)
        bins = self.bins
        range = self.range
        # bin dtype is not int
        self.assertRaises(
            TypeError, paddle.histogram_bin_edges, input, bins=1.5, range=range
        )
        # the range len is not equal 2
        self.assertRaises(
            ValueError,
            paddle.histogram_bin_edges,
            input,
            bins=bins,
            range=(0, 2, 3),
        )
        # the min of range greater than max
        self.assertRaises(
            ValueError,
            paddle.histogram_bin_edges,
            input,
            bins=bins,
            range=(3, 0),
        )


if __name__ == '__main__':
    unittest.main()
