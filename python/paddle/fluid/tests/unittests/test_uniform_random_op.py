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
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid


def output_hist(out):
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


def output_hist_diag(out):
    diag_num = min(out.shape)
    for i in range(diag_num):
        assert abs(out[i][i] - 1.0) < 1e-9
        # ignore diagonal elements
        out[i][i] = 100
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10
        }
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpWithDiagInit(TestUniformRandomOp):
    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            "diag_num": 784,
            "diag_step": 784,
            "diag_val": 1.0
        }
        self.output_hist = output_hist_diag


class TestUniformRandomOpSelectedRows(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()

        op = Operator(
            "uniform_random",
            Out="X",
            shape=[4, 784],
            min=-5.0,
            max=10.0,
            seed=10)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpSelectedRowsWithDiagInit(
        TestUniformRandomOpSelectedRows):
    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()

        op = Operator(
            "uniform_random",
            Out="X",
            shape=[4, 784],
            min=-5.0,
            max=10.0,
            seed=10,
            diag_num=4,
            diag_step=784,
            diag_val=1.0)
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist_diag(np.array(out.get_tensor()))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestUniformRandomOpApi(unittest.TestCase):
    def test_api(self):
        x = fluid.layers.data('x', shape=[16], dtype='float32', lod_level=1)
        y = fluid.layers.fc(x,
                            size=16,
                            param_attr=fluid.initializer.Uniform(
                                low=-0.5,
                                high=0.5,
                                seed=10,
                                diag_num=16,
                                diag_step=16,
                                diag_val=1.0))

        place = fluid.CPUPlace()
        x_tensor = fluid.create_lod_tensor(
            np.random.rand(3, 16).astype("float32"), [[1, 2]], place)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={'x': x_tensor}, fetch_list=[y], return_numpy=False)


if __name__ == "__main__":
    unittest.main()
