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
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

#paddle.enable_static()


class TestAddEqualDimOp(OpTest):
    def setUp(self):
        self.op_type = "addequaldim"
        self.inputs = {
            'X': np.random.random((20, 20)).astype(np.float64),
            'Y': np.random.random((20, 20)).astype(np.float64)
        }
        self.outputs = {'Out': (self.inputs['X'] + self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestAddEqualDimAPI(unittest.TestCase):
    #test paddle.tensor.addequaldim

    def setUp(self):
        self.shape = [20, 20]
        self.x = np.random.random((20, 20)).astype(np.float32)
        self.y = np.random.random((20, 20)).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            y = paddle.fluid.data('Y', self.shape)
            out = paddle.tensor.addequaldim(x, y)

            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x, 'Y': self.y}, fetch_list=[out])
        out_ref = self.x + self.y
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-04), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        y_tensor = paddle.to_tensor(self.y)
        out = paddle.tensor.addequaldim(x_tensor, y_tensor)
        out_ref = self.x + self.y
        self.assertEqual(np.allclose(out.numpy(), out_ref, rtol=1e-04), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.disable_static()
        x = np.random.random((20, 20)).astype(np.float64)
        y = np.random.random((10, 20)).astype(np.float64)
        self.assertRaises(Exception, paddle.tensor.addequaldim, x, y)
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], 'int32')
            y = paddle.fluid.data('Y', [10, 12], 'int32')
            self.assertRaises(TypeError, paddle.tensor.addequaldim, x, y)


if __name__ == "__main__":
    unittest.main()
