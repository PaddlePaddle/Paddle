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

import math
import unittest
import numpy as np
import paddle.fluid as fluid
from op_test import OpTest
from paddle import fluid
from paddle.fluid.layers import gru_unit
from paddle.fluid.framework import program_guard, Program


class TestGRUUnitAPIError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            D = 5
            layer = fluid.dygraph.nn.GRUUnit(size=D * 3)
            # the input must be Variable.
            x0 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.CPUPlace())
            self.assertRaises(TypeError, layer, x0)
            # the input dtype must be float32 or float64
            x = fluid.data(name='x', shape=[-1, D * 3], dtype='float16')
            hidden = fluid.data(name='hidden', shape=[-1, D], dtype='float32')
            self.assertRaises(TypeError, layer, x, hidden)


class GRUActivationType(OpTest):
    identity = 0
    sigmoid = 1
    tanh = 2
    relu = 3


def identity(x):
    return x


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return 2. * sigmoid(2. * x) - 1.


def relu(x):
    return np.maximum(x, 0)


class TestGRUUnitOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size = 5
            hidden_dim = 40
            input = fluid.data(name='input',
                               shape=[None, hidden_dim * 3],
                               dtype='float32')
            pre_hidden = fluid.data(name='pre_hidden',
                                    shape=[None, hidden_dim],
                                    dtype='float32')
            np_input = np.random.uniform(
                -0.1, 0.1, (batch_size, hidden_dim * 3)).astype('float64')
            np_pre_hidden = np.random.uniform(
                -0.1, 0.1, (batch_size, hidden_dim)).astype('float64')

            def test_input_Variable():
                gru_unit(np_input, pre_hidden, hidden_dim * 3)

            self.assertRaises(TypeError, test_input_Variable)

            def test_pre_hidden_Variable():
                gru_unit(input, np_pre_hidden, hidden_dim * 3)

            self.assertRaises(TypeError, test_pre_hidden_Variable)

            def test_input_type():
                error_input = fluid.data(name='error_input',
                                         shape=[None, hidden_dim * 3],
                                         dtype='int32')
                gru_unit(error_input, pre_hidden, hidden_dim * 3)

            self.assertRaises(TypeError, test_input_type)

            def test_pre_hidden_type():
                error_pre_hidden = fluid.data(name='error_pre_hidden',
                                              shape=[None, hidden_dim],
                                              dtype='int32')
                gru_unit(input, error_pre_hidden, hidden_dim * 3)

            self.assertRaises(TypeError, test_pre_hidden_type)


class TestGRUUnitOp(OpTest):
    batch_size = 5
    frame_size = 40
    activate = {
        GRUActivationType.identity: identity,
        GRUActivationType.sigmoid: sigmoid,
        GRUActivationType.tanh: tanh,
        GRUActivationType.relu: relu,
    }

    def set_inputs(self, origin_mode=False):
        batch_size = self.batch_size
        frame_size = self.frame_size
        self.op_type = 'gru_unit'
        self.inputs = {
            'Input':
            np.random.uniform(-0.1, 0.1,
                              (batch_size, frame_size * 3)).astype(self.dtype),
            'HiddenPrev':
            np.random.uniform(-0.1, 0.1,
                              (batch_size, frame_size)).astype(self.dtype),
            'Weight':
            np.random.uniform(-1. / math.sqrt(frame_size),
                              1. / math.sqrt(frame_size),
                              (frame_size, frame_size * 3)).astype(self.dtype),
        }
        self.attrs = {
            'activation': GRUActivationType.tanh,
            'gate_activation': GRUActivationType.sigmoid,
            'origin_mode': origin_mode
        }

    def set_outputs(self, origin_mode=False):
        # GRU calculations
        batch_size = self.batch_size
        frame_size = self.frame_size
        x = self.inputs['Input']
        h_p = self.inputs['HiddenPrev']
        w = self.inputs['Weight']
        b = self.inputs['Bias'] if 'Bias' in self.inputs else np.zeros(
            (1, frame_size * 3))
        g = x + np.tile(b, (batch_size, 1))
        w_u_r = w.flatten()[:frame_size * frame_size * 2].reshape(
            (frame_size, frame_size * 2))
        u_r = self.activate[self.attrs['gate_activation']](np.dot(h_p, w_u_r) +
                                                           g[:, :frame_size *
                                                             2])
        u = u_r[:, :frame_size]
        r = u_r[:, frame_size:frame_size * 2]
        r_h_p = r * h_p
        w_c = w.flatten()[frame_size * frame_size * 2:].reshape(
            (frame_size, frame_size))
        c = self.activate[self.attrs['activation']](np.dot(r_h_p, w_c) +
                                                    g[:, frame_size * 2:])
        g = np.hstack((u_r, c))
        if origin_mode:
            h = (1 - u) * c + u * h_p
        else:
            h = u * c + (1 - u) * h_p
        self.outputs = {
            'Gate': g.astype(self.dtype),
            'ResetHiddenPrev': r_h_p.astype(self.dtype),
            'Hidden': h.astype(self.dtype)
        }

    def setUp(self):
        self.dtype = 'float32' if fluid.core.is_compiled_with_rocm(
        ) else 'float64'
        self.set_inputs()
        self.set_outputs()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input', 'HiddenPrev', 'Weight'], ['Hidden'])


class TestGRUUnitOpOriginMode(TestGRUUnitOp):

    def setUp(self):
        self.dtype = 'float32' if fluid.core.is_compiled_with_rocm(
        ) else 'float64'
        self.set_inputs(origin_mode=True)
        self.set_outputs(origin_mode=True)


class TestGRUUnitOpWithBias(TestGRUUnitOp):

    def set_inputs(self, origin_mode=False):
        batch_size = self.batch_size
        frame_size = self.frame_size
        super(TestGRUUnitOpWithBias, self).set_inputs()
        self.inputs['Bias'] = np.random.uniform(
            -0.1, 0.1, (1, frame_size * 3)).astype(self.dtype)
        self.attrs = {
            'activation': GRUActivationType.identity,
            'gate_activation': GRUActivationType.sigmoid,
            'origin_mode': origin_mode
        }

    def test_check_grad(self):
        self.check_grad(['Input', 'HiddenPrev', 'Weight', 'Bias'], ['Hidden'])

    def test_check_grad_ingore_input(self):
        self.check_grad(['HiddenPrev', 'Weight', 'Bias'], ['Hidden'],
                        no_grad_set=set('Input'))


class TestGRUUnitOpWithBiasOriginMode(TestGRUUnitOpWithBias):

    def setUp(self):
        self.dtype = 'float32' if fluid.core.is_compiled_with_rocm(
        ) else 'float64'
        self.set_inputs(origin_mode=True)
        self.set_outputs(origin_mode=True)


if __name__ == '__main__':
    unittest.main()
