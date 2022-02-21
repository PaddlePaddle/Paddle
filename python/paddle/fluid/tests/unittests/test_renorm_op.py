# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.set_device('cpu')


class TestRenormAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]])
        self.p = 1.0
        self.dim = 2
        self.max_norm = 2.05

    def test_renorm_api(self):
        paddle.enable_static()
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            #x = fluid.layers.data(name = 'x',shape=[-1, 2, 3])
            x = paddle.static.data(name="x", shape=[-1, 2, 3], dtype='float64')
            z = paddle.renorm(x, self.p, self.dim, self.max_norm)
            exe = fluid.Executor(fluid.CPUPlace())
            res, = exe.run(feed={"x": self.data_x},
                           fetch_list=[z],
                           return_numpy=False)
        expected = np.array([[[0.40594056, 0.29285714, -0.41000000],
                              [0.60891086, 0.04392857, 0.61500001]],
                             [[0.40594056, -1.17142856, 0.41000000],
                              [0.62920785, 0.54178572, 0.61500001]]])
        self.assertTrue(np.allclose(expected, np.array(res)))

    def test_dygraph_api(self):
        self.input_data()
        # case axis none
        with fluid.dygraph.guard():
            input = [[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]]
            x = paddle.to_tensor(input, stop_gradient=False)
            y = paddle.renorm(x, 1.0, 2, 2.05)
            expected = np.array([[[0.40594056, 0.29285714, -0.41000000],
                                  [0.60891086, 0.04392857, 0.61500001]],
                                 [[0.40594056, -1.17142856, 0.41000000],
                                  [0.62920785, 0.54178572, 0.61500001]]])
            self.assertTrue(np.allclose(expected, np.array(y)))
            z = paddle.mean(y)
            z.backward(retain_graph=True)
            expected_grad = np.array(
                [[[0, 0.01394558, 0.02733333], [0, 0.01394558, 0.00683333]],
                 [[0, 0.01045918, 0.00683333], [0, 0.01394558, 0.00683333]]])
            self.assertTrue(np.allclose(expected_grad, np.array(x.grad)))
        #test exception:
        with fluid.dygraph.guard():
            input = [[[2.0, 2, -2], [3, 0.3, 3]], [[2, -8, 2], [3.1, 3.7, 3]]]
            x = paddle.to_tensor(input, stop_gradient=False)
            exp = False
            try:
                paddle.renorm(x, 1.0, 8, 2.05)
            except:
                exp = True
            self.assertTrue(exp)
            exp = False
            try:
                paddle.renorm(x, 1.0, -4, 2.05)
            except:
                exp = True
            self.assertTrue(exp)
            y = paddle.renorm(x, 1.0, -1, 2.05)
            expected = np.array([[[0.40594056, 0.29285714, -0.41000000],
                                  [0.60891086, 0.04392857, 0.61500001]],
                                 [[0.40594056, -1.17142856, 0.41000000],
                                  [0.62920785, 0.54178572, 0.61500001]]])
            self.assertTrue(np.allclose(expected, np.array(y)))


if __name__ == '__main__':
    unittest.main()
