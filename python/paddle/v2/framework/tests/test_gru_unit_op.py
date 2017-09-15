import math
import unittest
import numpy as np
from op_test import OpTest


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))


def tanh_np(x):
    return 2. * sigmoid_np(2. * x) - 1.


class TestGRUUnitOp(OpTest):
    def setUp(self):
        batch_size = 3
        frame_size = 5
        self.op_type = "gru_unit"
        self.inputs = {
            'input': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size * 3)).astype("float32"),
            'hidden_prev': np.random.uniform(
                -0.1, 0.1, (batch_size, frame_size)).astype("float32"),
            'weight': np.random.uniform(
                -1. / math.sqrt(frame_size), 1. / math.sqrt(frame_size),
                (frame_size, frame_size * 3)).astype("float32"),
            'bias': np.random.uniform(-0.1, 0.1,
                                      (1, frame_size * 3)).astype("float32")
        }
        x = self.inputs['input']
        h_p = self.inputs['hidden_prev']
        w = self.inputs['weight']
        b = self.inputs['bias']
        g = x + np.tile(b, (batch_size, 1))
        w_u_r = w.flatten()[:frame_size * frame_size * 2].reshape(
            (frame_size, frame_size * 2))
        u_r = sigmoid_np(np.dot(h_p, w_u_r) + g[:, :frame_size * 2])
        u = u_r[:, :frame_size]
        r = u_r[:, frame_size:frame_size * 2]
        r_h_p = r * h_p
        w_c = w.flatten()[frame_size * frame_size * 2:].reshape(
            (frame_size, frame_size))
        c = tanh_np(np.dot(r_h_p, w_c) + g[:, frame_size * 2:])
        g = np.hstack((u_r, c))
        h = u * h_p + (1 - u) * c
        self.outputs = {'gate': g, 'reset_hidden_prev': r_h_p, 'hidden': h}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['input', 'hidden_prev', 'weight', 'bias'], ['hidden'],
            max_relative_error=0.007)


if __name__ == '__main__':
    unittest.main()
