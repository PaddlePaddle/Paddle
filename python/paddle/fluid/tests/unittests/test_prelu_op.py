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


class TestPReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.prelu, 0.1, 'all')
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.prelu, x_int32, 'all')
            # support the input dtype is float32
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float32')
            fluid.layers.prelu(x_fp16, 'all')


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


def prelu_t(x, mode, param_attr=None, name=None):
    helper = fluid.layer_helper.LayerHelper('prelu', **locals())
    alpha_shape = [1, x.shape[1], 1, 1]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype='float32',
        is_bias=False,
        default_initializer=fluid.initializer.ConstantInitializer(0.25))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                'Alpha': alpha},
        attrs={"mode": mode},
        outputs={"Out": out})
    return out


# error message test if mode is not one of 'all', 'channel', 'element'
class TestModeError(unittest.TestCase):
    def test_mode_error(self):
        main_program = Program()
        with fluid.program_guard(main_program, Program()):
            x = fluid.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'any')
            except Exception as e:
                assert (e.args[0].find('InvalidArgumentError') != -1)


if __name__ == "__main__":
    unittest.main()
