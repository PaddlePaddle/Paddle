import math
import unittest
import numpy as np
from op_test import OpTest


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


class TestGRUUnitOp(OpTest):
    batch_size = 5
    frame_size = 10
    activate = {
        GRUActivationType.identity: identity,
        GRUActivationType.sigmoid: sigmoid,
        GRUActivationType.tanh: tanh,
        GRUActivationType.relu: relu,
    }

    def set_inputs(self):
        batch_size = self.batch_size
        frame_size = self.frame_size
        self.op_type = 'gru_unit'
        self.inputs = {
            'Input': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size * 3)).astype('float64'),
            'HiddenPrev': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size)).astype('float64'),
            'Weight': np.random.uniform(
                -1. / math.sqrt(frame_size), 1. / math.sqrt(frame_size),
                (frame_size, frame_size * 3)).astype('float64'),
        }
        self.attrs = {
            'activation': GRUActivationType.tanh,
            'gate_activation': GRUActivationType.sigmoid
        }

    def set_outputs(self):
        # GRU calculations
        batch_size = self.batch_size
        frame_size = self.frame_size
        x = self.inputs['Input']
        h_p = self.inputs['HiddenPrev']
        w = self.inputs['Weight']
        b = self.inputs['Bias'] if self.inputs.has_key('Bias') else np.zeros(
            (1, frame_size * 3))
        g = x + np.tile(b, (batch_size, 1))
        w_u_r = w.flatten()[:frame_size * frame_size * 2].reshape(
            (frame_size, frame_size * 2))
        u_r = self.activate[self.attrs['gate_activation']](np.dot(
            h_p, w_u_r) + g[:, :frame_size * 2])
        u = u_r[:, :frame_size]
        r = u_r[:, frame_size:frame_size * 2]
        r_h_p = r * h_p
        w_c = w.flatten()[frame_size * frame_size * 2:].reshape(
            (frame_size, frame_size))
        c = self.activate[self.attrs['activation']](np.dot(r_h_p, w_c) +
                                                    g[:, frame_size * 2:])
        g = np.hstack((u_r, c))
        h = u * c + (1 - u) * h_p
        self.outputs = {
            'Gate': g.astype('float64'),
            'ResetHiddenPrev': r_h_p.astype('float64'),
            'Hidden': h.astype('float64')
        }

    def setUp(self):
        self.set_inputs()
        self.set_outputs()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input', 'HiddenPrev', 'Weight'], ['Hidden'])


class TestGRUUnitOpWithBias(TestGRUUnitOp):
    def set_inputs(self):
        batch_size = self.batch_size
        frame_size = self.frame_size
        super(TestGRUUnitOpWithBias, self).set_inputs()
        self.inputs['Bias'] = np.random.uniform(
            -0.1, 0.1, (1, frame_size * 3)).astype('float64')
        self.attrs = {
            'activation': GRUActivationType.identity,
            'gate_activation': GRUActivationType.sigmoid
        }

    def test_check_grad(self):
        self.check_grad(['Input', 'HiddenPrev', 'Weight', 'Bias'], ['Hidden'])

    def test_check_grad_ingore_input(self):
        self.check_grad(
            ['HiddenPrev', 'Weight', 'Bias'], ['Hidden'],
            no_grad_set=set('Input'))


if __name__ == '__main__':
    unittest.main()
