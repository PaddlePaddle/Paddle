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
import paddle.fluid as fluid
import six
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci


class TestPReluAPIError(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            layer = fluid.PRelu(
                mode='all',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(1.0)))
            # the input must be Variable.
            x0 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
            self.assertRaises(TypeError, layer, x0)
            # the input dtype must be float32
            data_t = fluid.data(
                name="input", shape=[5, 200, 100, 100], dtype="float64")
            self.assertRaises(TypeError, layer, data_t)


class PReluTest(OpTest):
    def setUp(self):
        self.init_input_shape()
        self.init_attr()
        self.op_type = "prelu"

        x_np = np.random.uniform(-1, 1, self.x_shape)
        # Since zero point in prelu is not differentiable, avoid randomize
        # zero.
        x_np[np.abs(x_np) < 0.005] = 0.02

        if self.attrs == {'mode': "all"}:
            alpha_np = np.random.uniform(-1, -0.5, (1))
        elif self.attrs == {'mode': "channel"}:
            alpha_np = np.random.uniform(-1, -0.5, (1, x_np.shape[1], 1, 1))
        else:
            alpha_np = np.random.uniform(-1, -0.5, \
                (1, x_np.shape[1], x_np.shape[2], x_np.shape[3]))
        self.inputs = {'X': x_np, 'Alpha': alpha_np}

        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'],
                                     0.) * self.inputs['Alpha']
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def init_input_shape(self):
        self.x_shape = (2, 100, 3, 4)

    def init_attr(self):
        self.attrs = {'mode': "channel"}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Alpha'], 'Out')


# TODO(minqiyang): Resume these test cases after fixing Python3 CI job issues
if six.PY2:

    @skip_check_grad_ci(
        reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
    )
    class TestModeAll(PReluTest):
        def init_input_shape(self):
            self.x_shape = (2, 3, 4, 5)

        def init_attr(self):
            self.attrs = {'mode': "all"}

    class TestModeElt(PReluTest):
        def init_input_shape(self):
            self.x_shape = (3, 2, 5, 10)

        def init_attr(self):
            self.attrs = {'mode': "element"}


class TestPReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.prelu, 1, 'all')
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.prelu, x_int32, 'all')
            # support the input dtype is float32
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float32')
            fluid.layers.prelu(x_fp16, 'all')


if __name__ == "__main__":
    unittest.main()
