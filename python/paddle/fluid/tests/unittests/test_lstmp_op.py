#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
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
        proj_clip = 0.0,
        cell_clip = 0.0,
        act_gate=None,
        act_cell=None,
        act_cand=None,
        act_proj=None):
    def _step(x, w_r, w_rh, w_c, r_pre, c_pre, proj_clip, cell_clip, act_gate, act_cell, act_cand,
              act_proj):
        #import pdb; pdb.set_trace()
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

        def array_clip(a, clip):
            #print('clip:{}'.format(clip))
            #print('old' + str(a))

            size = np.prod(a.shape)
            new_a = np.reshape(a, (size))
	    for i in range(size):
		new_a[i] = max(new_a[i], -1.0 * clip)
		new_a[i] = min(new_a[i], clip)
            new_a = np.reshape(new_a, a.shape)
            #print('new' + str(new_a))
            return new_a

        if cell_clip > 0.0:
            c = array_clip(c, cell_clip)
        if w_c is None:
            g_o = act_gate(g_o)  # 1 x D
        else:
            _, _, w_oc = np.split(w_c, 3, axis=1)
            g_o = act_gate(g_o + w_oc * c)  # 1 x D
        h = g_o * act_cell(c)
        # projection
        r = np.dot(h, w_rh)
        r = act_proj(r)
        if proj_clip > 0.0:
            r = array_clip(r, proj_clip)
        return r, c

    def _reverse(x, offset):
        y = np.zeros_like(x)
        for i in range(len(offset) - 1):
            b, e = offset[i], offset[i + 1]
            y[b:e, :] = np.flip(x[b:e, :], 0)
        return y

    offset = [0]
    for l in lod[0]:
        offset.append(offset[-1] + l)
    batch_size = len(lod[0])
    # recurrent projection state
    projection = []
    cell = []
    input = _reverse(input, offset) if is_reverse else input
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        # compute one sequence
        seq_len = lod[0][i]
        x = input[offset[i]:offset[i + 1], :]
        #r_pre = np.dot(h0[i], w_rh)  # 1 x P
        r_pre = h0[i]
        #r_pre = act_proj(r_pre)
        c_pre = c0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            r_pre, c_pre = _step(x[j], w_r, w_rh, w_c, r_pre, c_pre, proj_clip, cell_clip, act_gate,
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

    def setUp2(self):
        self.set_argument()
        # projection size
        self.P = 2

        self.reset_argument()
        self.op_type = 'lstmp'
        self.act_proj = 'identity'
        self.use_peepholes = False
        self.has_initial_state = True
        self.lod=[[5]]

        T = sum(self.lod[0])
        N = len(self.lod[0])

        proj_clip = 0.5
        cell_clip = 0.0

        #import pdb; pdb.set_trace()
        x=np.array([[-0.50806344, 0.50909436], \
	 [-0.50087136, 0.4904187 ], \
	 [-0.48933774, 0.50408053], \
	 [ 0.00896523, 0.00770854], \
	 [-0.00851139,-0.01005108]])
        wx = np.array([[ 0.2932311,  -0.8829277,   1.100133,    0.8197811,  -0.8194872,  -0.829262, 0.7708865,  -0.62339246, -0.7656475,   0.4283645,  -0.27164033, -0.3600223 ], \
            [-0.609142,    0.25025278,  0.15731744, -0.66051376, -0.70994514,  0.8344964, -0.00551117, -0.7072167,  -0.63929003, -0.52340907, -0.8842589,   0.9531688 ]])
        x=np.dot(x, wx)

        w = np.array([[ 0.7808204, -0.7412322,  -0.9458036,  -0.01664658,  0.7930616,   0.10208707, 0.20036687, -0.16743736,  1.0295134,  -0.3118722,   0.02241168,  0.3154219 ], \
 [-0.29026014,  0.24638331, -0.5435432,   0.87635124, -0.96091515, -0.1411362, 0.58606523, -0.38996056, -0.9003789,   0.8540163,  -0.8831781,  -0.28499633]])

        w_rh=np.array([[ 0.15685119,  0.05694652], [-0.9641068,  -1.5106804 ], [ 0.3599193,   1.2540514 ]])	
        w_b = np.array([[-0.49999997,  0.5,        -0.49999997, -0.5,         0.5,         0.5, 0.49999997, -0.49999997,  0.49999997, -0.5,         0.49999997,  0.5       ]])
        h0 = np.array([[-1.3392334e-04, -6.8468950e-04]])
        c0 = np.array([[4.5552300e-04,  1.3302206e-03, -3.6721351e-04]])
        w_c=None
        self.lod=[[5]]
        #import pdb; pdb.set_trace()
        r, c = lstmp(x, self.lod, h0, c0, w, w_rh, w_b, w_c, self.is_reverse, proj_clip, cell_clip,
                     ACTIVATION[self.act_gate], ACTIVATION[self.act_cell],
                     ACTIVATION[self.act_cand], ACTIVATION[self.act_proj])
        self.inputs = {'Input': (x, self.lod), 'Weight': w, 'ProjWeight': w_rh}

        self.inputs['Bias'] = w_b

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
            'proj_clip':proj_clip,
            'cell_clip':cell_clip,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'proj_activation': self.act_proj
        }

    def setUp(self):
        self.set_argument()
        # projection size
        self.P = 10
        #self.D = 9
        self.act_proj = self.act_cell

        self.reset_argument()
        self.op_type = 'lstmp'
        #self.use_peepholes=False
        #self.lod=[[7]]
        #self.act_proj='identity'
        #self.act_proj='tanh'

        T = sum(self.lod[0])
        N = len(self.lod[0])
        #np.random.seed=123
        x = np.random.normal(size=(T, 4 * self.D)).astype('float64')
        if self.has_initial_state:
            h0 = np.random.normal(size=(N, self.P)).astype('float64')
            c0 = np.random.normal(size=(N, self.D)).astype('float64')
        else:
            h0 = np.zeros((N, self.P)).astype('float64')
            c0 = np.zeros((N, self.D)).astype('float64')
        w = np.random.normal(size=(self.P, 4 * self.D)).astype('float64')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float64')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float64')

        w_b = b[:, 0:4 * self.D]
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        w_rh = np.random.normal(size=(self.D, self.P)).astype('float64')
        proj_clip = 0.1
        cell_clip = 0.1
        #import pdb; pdb.set_trace()
        r, c = lstmp(x, self.lod, h0, c0, w, w_rh, w_b, w_c, self.is_reverse, proj_clip, cell_clip,
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
            'proj_clip':proj_clip,
            'cell_clip':cell_clip,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'proj_activation': self.act_proj
        }

    def test_check_output(self):
        self.check_output(atol=1e-8)

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005)


class TestLstmpOpHasInitial(TestLstmpOp):
    def reset_argument(self):
        self.has_initial_state = True

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'H0', 'C0'],
            ['Projection'], numeric_grad_delta=0.0000005,
            max_relative_error=1e-2)

    def test_check_grad_ingore_bias(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'ProjWeight', 'Weight'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005,
            no_grad_set=set('Bias'))

    def test_check_grad_ingore_weight(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005,
            no_grad_set=set('Weight'))

    def test_check_grad_ingore_proj_weight(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005,
            no_grad_set=set('ProjWeight'))

    def test_check_grad_ingore_input(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Weight', 'ProjWeight', 'Bias'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005, 
            no_grad_set=set('Input'))

    def test_check_grad_ingore_h0(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'C0'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005,
            no_grad_set=set('H0'))

    def test_check_grad_ingore_c0(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'ProjWeight', 'Bias', 'H0'], ['Projection'],
            max_relative_error=1e-2, numeric_grad_delta=0.0000005,
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
