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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid.framework as framework


def output_hist(out, dtype):
    hist, _ = np.histogram(out, range=(-5, 10))
    if dtype == np.float16:
        hist = hist.astype(np.float32)
        hist /= float(out.size)
        hist = hist.astype(np.float16)
    else:
        hist = hist.astype(dtype)
        hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.dtype = np.float32
        self.shape = [1000, 784]
        self.init_dtype()

        self.inputs = {}
        self.attrs = {
            "shape": self.shape,
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            "dtype": framework.convert_np_dtype_to_dtype_(self.dtype)
        }
        self.outputs = {"Out": np.zeros(self.shape).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]), self.dtype)
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestFP16UniformRandomOp(TestUniformRandomOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestUniformRandomOpSelectedRows(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.init_dtype()

    def init_dtype(self):
        pass

    def get_places(self):
        if self.dtype == np.float16:
            places = []
        else:
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
            seed=10,
            dtype=int(framework.convert_np_dtype_to_dtype_(self.dtype)))
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [4, 784])
        hist, prob = output_hist(np.array(out.get_tensor()), self.dtype)
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


class TestFP16UniformRandomOpSelectedRows(TestUniformRandomOpSelectedRows):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
