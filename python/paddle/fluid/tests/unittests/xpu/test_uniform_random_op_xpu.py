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

import sys
sys.path.append("..")
import subprocess
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()


def output_hist(out):
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestXPUUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.,
            "max": 10.0,
            "seed": 10,
            "use_xpu": True
        }
        self.output_hist = output_hist

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            outs = self.calc_output(place)
            outs = [np.array(out) for out in outs]
            outs.sort(key=len)
            self.verify_output(outs)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestXPUUniformRandomOpSelectedRows(unittest.TestCase):
    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        paddle.manual_seed(10)
        op = Operator(
            "uniform_random",
            Out="X",
            shape=[1000, 784],
            min=-5.0,
            max=10.0,
            seed=10)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


if __name__ == "__main__":
    unittest.main()
