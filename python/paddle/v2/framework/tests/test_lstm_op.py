import unittest
import numpy as np
from op_test import OpTest


def identity(x):
    return x


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return 2. * sigmoid(2. * x) - 1.


def relu(x):
    return np.maximum(x, 0)


def lstm(
        input,  # T x 4D
        lod,  # 1 x N
        h0=None,  # N x D
        c0=None,  # N x D
        w_h=None,  # D x 4D
        w_b=None,  # 1 x 4D
        w_c=None,  # 1 x 3D
        is_reverse=False,
        gate_act=None,
        cell_act=None,
        cand_act=None):
    def _step(x, w_h, w_c, h_pre, c_pre, gate_act, cell_act, cand_act):
        g = np.dot(h_pre, w_h)  # 1 x 4D
        g = g + x
        g = np.reshape(g, (1, g.size))
        c, g_i, g_f, g_o = np.split(g, 4, axis=1)
        if w_c is None:
            g_i = gate_act(g_i)  # 1 x D
            g_f = gate_act(g_f)  # 1 x D
        else:
            w_ic, w_fc, w_oc = np.split(w_c, 3, axis=1)
            g_i = gate_act(g_i + w_ic * c_pre)  # 1 x D
            g_f = gate_act(g_f + w_fc * c_pre)  # 1 x D
        c = g_f * c_pre + g_i * cand_act(c)  # 1 x D

        if w_c is None:
            g_o = gate_act(g_o)  # 1 x D
        else:
            _, _, w_oc = np.split(w_c, 3, axis=1)
            g_o = gate_act(g_o + w_oc * c)  # 1 x D
        h = g_o * cell_act(c)
        return h, c

    offset = lod[0]
    batch_size = len(offset) - 1
    hidden = []
    cell = []
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        # compute one sequence
        seq_len = offset[i + 1] - offset[i]
        x = input[offset[i]:offset[i + 1], :]
        h_pre = h0[i]  # 1 x D
        c_pre = h0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            h_pre, c_pre = _step(x[j], w_h, w_c, h_pre, c_pre, gate_act,
                                 cell_act, cand_act)
            hidden.append(h_pre.flatten())
            cell.append(c_pre.flatten())

    hidden = np.array(hidden).astype("float64")
    cell = np.array(cell).astype("float64")
    assert hidden.shape == (input.shape[0], input.shape[1] / 4)
    assert cell.shape == (input.shape[0], input.shape[1] / 4)
    return hidden, cell


class LstmUnitTest(OpTest):
    def set_data(self):
        lod = [[0, 2, 6, 9]]
        shape = (9, 64)

        x = np.random.normal(size=(9, 4 * 64)).astype("float64")
        h0 = np.random.normal(size=(4, 64)).astype("float64")
        c0 = np.random.normal(size=(4, 64)).astype("float64")
        w = np.random.normal(size=(64, 4 * 64)).astype("float64")
        b = np.random.normal(size=(1, 7 * 64)).astype("float64")

        w_b = b[:, 4 * 64]
        w_c = b[:, 4 * 64:]
        h, c = lstm(x, lod, h0, c0, w, w_b, w_c, False, sigmoid, tanh, tanh)

        self.inputs = {'Input': x, 'H0': h0, 'C0': c0, 'Weight': w, 'Bias': b}
        self.inputs = {'Hidden': h, 'Cell': c}
        self.attrs = {
            'usePeepholes': True,
            'isReverse': False,
            'gateActivation': 'sigmoid',
            'cellActivation': 'tanh',
            'candidateActivation': 'tanh'
        }

    def setUp(self):
        self.set_data()
        self.op_type = "lstm"

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
