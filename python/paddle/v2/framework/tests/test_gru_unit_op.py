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
    activate = {
        GRUActivationType.identity: identity,
        GRUActivationType.sigmoid: sigmoid,
        GRUActivationType.tanh: tanh,
        GRUActivationType.relu: relu,
    }

    def setUp(self):
        batch_size = 3
        frame_size = 5
        self.op_type = 'gru_unit'
        self.inputs = {
            'Input': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size * 3)).astype('float32'),
            'HiddenPrev': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size)).astype('float32'),
            'Weight': np.random.uniform(
                -1. / math.sqrt(frame_size), 1. / math.sqrt(frame_size),
                (frame_size, frame_size * 3)).astype('float32'),
            'Bias': np.random.uniform(-0.1, 0.1,
                                      (1, frame_size * 3)).astype('float32')
        }
        self.attrs = {
            'activation': GRUActivationType.tanh,
            'gate_activation': GRUActivationType.sigmoid
        }
        # GRU calculations
        x = self.inputs['Input']
        h_p = self.inputs['HiddenPrev']
        w = self.inputs['Weight']
        b = self.inputs['Bias']
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
        h = u * h_p + (1 - u) * c

        self.outputs = {'Gate': g, 'ResetHiddenPrev': r_h_p, 'Hidden': h}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['Input', 'HiddenPrev', 'Weight', 'Bias'], ['Hidden'],
            max_relative_error=0.007)


if __name__ == '__main__':
    unittest.main()
