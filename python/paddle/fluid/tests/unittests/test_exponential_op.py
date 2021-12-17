#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
from op_test import OpTest

paddle.enable_static()


class TestExponentialOp1(OpTest):
    def setUp(self):
        self.op_type = "exponential"
        self.config()

        self.attrs = {"lambda": self.lam}
        self.inputs = {'X': np.empty([2048, 1024], dtype=self.dtype)}
        self.outputs = {'Out': np.ones([2048, 1024], dtype=self.dtype)}

    def config(self):
        self.lam = 0.5
        self.dtype = "float64"

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist1, _ = np.histogram(outs[0], range=(0, 5))
        hist1 = hist1.astype("float32")
        hist1 = hist1 / float(outs[0].size)

        data_np = np.random.exponential(1. / self.lam, [2048, 1024])
        hist2, _ = np.histogram(data_np, range=(0, 5))
        hist2 = hist2.astype("float32")
        hist2 = hist2 / float(data_np.size)

        print((hist1 - hist2) / hist1)
        self.assertTrue(
            np.allclose(
                hist1, hist2, rtol=0.01),
            "actual: {}, expected: {}".format(hist1, hist2))

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[np.zeros(
                [2048, 1024], dtype=self.dtype)],
            user_defined_grad_outputs=[
                np.random.rand(2048, 1024).astype(self.dtype)
            ])


class TestExponentialOp2(TestExponentialOp1):
    def config(self):
        self.lam = 0.25
        self.dtype = "float32"


class TestExponentialAPI(unittest.TestCase):
    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x_np = np.full([10, 10], -1.)
            x = paddle.static.data(name="X", shape=[10, 10], dtype='float64')
            x.exponential_(1.0)

            exe = paddle.static.Executor(paddle.CPUPlace())
            out = exe.run(paddle.static.default_main_program(),
                          feed={"X": x_np},
                          fetch_list=[x])
            self.assertTrue(np.min(out) >= 0)

    def test_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.full([10, 10], -1., dtype='float32')
        x.exponential_(0.5)
        self.assertTrue(np.min(x.numpy()) >= 0)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
