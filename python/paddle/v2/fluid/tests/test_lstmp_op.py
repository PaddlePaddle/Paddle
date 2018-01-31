#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import unittest
import numpy as np
import test_lstm_op as LstmTest

ACTIVATION = {
    'identity': LstmTest.identity,
    'sigmoid': LstmTest.sigmoid,
    'tanh': LstmTest.tanh,
    'relu': LstmTest.relu
}


# LSTM with recurrent projection Layer
def lstmp(
        input,  # T x 4D
        lod,  # 1 x N
        h0=None,  # N x D
        c0=None,  # N x D
        w_r=None,  # P x 4D
        w_rh=None,  # D x P
        w_b=None,  # 1 x 4D
        w_c=None,  # 1 x 3D
        is_reverse=False,
        act_gate=None,
        act_cell=None,
        act_cand=None,
        act_proj=None):
    def _step(x, w_r, w_rh, w_c, r_pre, c_pre, act_gate, act_cell, act_cand,
              act_proj):
        g = np.dot(r_pre, w_r)  # 1 x 4D
        g = g + x
        g = np.reshape(g, (1, g.size))
        c, g_i, g_f, g_o = np.split(g, 4, axis=1)
        if w_c is None:
            g_i = act_gate(g_i)  # 1 x D
            g_f = act_gate(g_f)  # 1 x D
        else:
            w_ic, w_fc, _ = np.split(w_c, 3, axis=1)
            g_i = act_gate(g_i + w_ic * c_pre)  # 1 x D
            g_f = act_gate(g_f + w_fc * c_pre)  # 1 x D
        c = g_f * c_pre + g_i * act_cand(c)  # 1 x D

        if w_c is None:
            g_o = act_gate(g_o)  # 1 x D
        else:
            _, _, w_oc = np.split(w_c, 3, axis=1)
            g_o = act_gate(g_o + w_oc * c)  # 1 x D
        h = g_o * act_cell(c)
        # projection
        r = np.dot(h, w_rh)
        r = act_proj(r)
        return r, c

    def _reverse(x, lod):
        y = np.zeros_like(x)
        for i in range(len(lod) - 1):
            b, e = lod[i], lod[i + 1]
            y[b:e, :] = np.flip(x[b:e, :], 0)
        return y

    offset = lod[0]
    batch_size = len(offset) - 1
    # recurrent projection state
    projection = []
    cell = []
    input = _reverse(input, offset) if is_reverse else input
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        # compute one sequence
        seq_len = offset[i + 1] - offset[i]
        x = input[offset[i]:offset[i + 1], :]
        r_pre = np.dot(h0[i], w_rh)  # 1 x P
        r_pre = act_proj(r_pre)
        c_pre = c0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            r_pre, c_pre = _step(x[j], w_r, w_rh, w_c, r_pre, c_pre, act_gate,
                                 act_cell, act_cand, act_proj)
            projection.append(r_pre.flatten())
            cell.append(c_pre.flatten())

    projection = np.array(projection).astype('float64')
    cell = np.array(cell).astype('float64')

    projection = _reverse(projection, offset) if is_reverse else projection
    cell = _reverse(cell, offset) if is_reverse else cell

    assert projection.shape == (input.shape[0], w_r.shape[0])  # T x P
    assert cell.shape == (input.shape[0], input.shape[1] / 4)  # T x D
    return projection, cell


class TestLstmpOp(LstmTest.TestLstmOp):
    def reset_argument(self):
        pass

    def setUp(self):
        self.set_argument()
        # projection size
        self.P = 10
        self.act_proj = self.act_cell

        self.reset_argument()
        self.op_type = 'lstmp'

        T = self.lod[0][-1]
        N = len(self.lod[0]) - 1

        x = np.random.normal(size=(T, 4 * self.D)).astype('float64')
        if self.has_initial_state:
            h0 = np.random.normal(size=(N, self.D)).astype('float64')
            c0 = np.random.normal(size=(N, self.D)).astype('float64')
        else:
            h0 = np.zeros((N, self.D)).astype('float64')
            c0 = np.zeros((N, self.D)).astype('float64')
        w = np.random.normal(size=(self.P, 4 * self.D)).astype('float64')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float64')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float64')

        w_b = b[:, 0:4 * self.D]
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        w_rh = np.random.normal(size=(self.D, self.P)).astype('float64')
        r, c = lstmp(x, self.lod, h0, c0, w, w_rh, w_b, w_c, self.is_reverse,
                     ACTIVATION[self.act_gate], ACTIVATION[self.act_cell],
                     ACTIVATION[self.act_cand], ACTIVATION[self.act_proj])

        self.inputs = {'Input': (x, self.lod), 'Weight': w, 'ProjWeight': w_rh}

        self.inputs['Bias'] = b

        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0

        self.outputs = {
            'Projection': (r, self.lod),
            'Cell': (c, self.lod),
        }
        self.attrs = {
            'use_peepholes': self.use_peepholes,
            'is_reverse': self.is_reverse,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'proj_activation': self.act_proj
        }

    def test_check_output(self):
        self.check_output(atol=1e-8)

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2)


class TestLstmpOpHasInitial(TestLstmpOp):
    def reset_argument(self):
        self.has_initial_state = True

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'H0', 'C0'],
            ['Projection'],
            max_relative_error=1e-2)

    def test_check_grad_ingore_bias(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'ProjWeight', 'Weight'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('Bias'))

    def test_check_grad_ingore_weight(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('Weight'))

    def test_check_grad_ingore_proj_weight(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('ProjWeight'))

    def test_check_grad_ingore_input(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Weight', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('Input'))

    def test_check_grad_ingore_h0(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'C0'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('H0'))

    def test_check_grad_ingore_c0(self):
        N = len(self.lod[0]) - 1
        self.outputs['OrderedP0'] = np.zeros((N, self.P)).astype('float64')
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'H0'], ['Projection'],
            max_relative_error=1e-2,
            no_grad_set=set('C0'))


class TestLstmpOpRerverse(TestLstmpOp):
    def reset_argument(self):
        self.is_reverse = True


class TestLstmpOpNotUsePeepholes(TestLstmpOp):
    def reset_argument(self):
        self.use_peepholes = False


class TestLstmpOpLinearProjection(TestLstmpOp):
    def reset_argument(self):
        self.act_proj = 'identity'


if __name__ == '__main__':
    unittest.main()
