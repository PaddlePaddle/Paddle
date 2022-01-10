#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from op_test import OpTest
import numpy as np


def output_hist(out):
    hist, _ = np.histogram(out, bins=2)
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.5 * np.ones((2))
    return hist, prob


class TestBernoulliOp(OpTest):
    def setUp(self):
        self.op_type = "bernoulli"
        self.inputs = {"X": np.random.uniform(size=(1000, 784))}
        self.attrs = {}
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestBernoulliApi(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([1024, 1024])
        out = paddle.bernoulli(x)
        paddle.enable_static()
        hist, prob = output_hist(out.numpy())
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))

    def test_static(self):
        x = paddle.rand([1024, 1024])
        out = paddle.bernoulli(x)
        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(paddle.static.default_main_program(),
                      fetch_list=[out.name])
        hist, prob = output_hist(out[0])
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


if __name__ == "__main__":
    unittest.main()
