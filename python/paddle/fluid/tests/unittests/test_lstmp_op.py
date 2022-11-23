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

import unittest
import numpy as np
import test_lstm_op as LstmTest
from paddle import fluid
from paddle.fluid import Program, program_guard

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
        proj_clip=0.0,
        cell_clip=0.0,
        act_gate=None,
        act_cell=None,
        act_cand=None,
        act_proj=None):

    def _step(x, w_r, w_rh, w_c, r_pre, c_pre, proj_clip, cell_clip, act_gate,
              act_cell, act_cand, act_proj):
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
            size = np.prod(a.shape)
            new_a = np.reshape(a, (size))
            for i in range(size):
                new_a[i] = max(new_a[i], -1.0 * clip)
                new_a[i] = min(new_a[i], clip)
            new_a = np.reshape(new_a, a.shape)
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
        r_pre = h0[i]
        c_pre = c0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            r_pre, c_pre = _step(x[j], w_r, w_rh, w_c, r_pre, c_pre, proj_clip,
                                 cell_clip, act_gate, act_cell, act_cand,
                                 act_proj)
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

        T = sum(self.lod[0])
        N = len(self.lod[0])
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
        r, c = lstmp(x, self.lod, h0, c0, w, w_rh, w_b, w_c, self.is_reverse,
                     proj_clip, cell_clip, ACTIVATION[self.act_gate],
                     ACTIVATION[self.act_cell], ACTIVATION[self.act_cand],
                     ACTIVATION[self.act_proj])

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
            'proj_clip': proj_clip,
            'cell_clip': cell_clip,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand,
            'proj_activation': self.act_proj
        }

    def test_check_output(self):
        self.check_output(atol=1e-8, check_dygraph=False)

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'Weight', 'ProjWeight', 'Bias'],
                        ['Projection'],
                        numeric_grad_delta=0.0000005,
                        check_dygraph=False)


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
        self.check_grad(['Input', 'Weight', 'ProjWeight', 'Bias', 'H0', 'C0'],
                        ['Projection'],
                        numeric_grad_delta=0.0000005,
                        check_dygraph=False)

    def test_check_grad_ingore_bias(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'ProjWeight', 'Weight'], ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('Bias'),
                        check_dygraph=False)

    def test_check_grad_ingore_weight(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'ProjWeight', 'Bias'], ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('Weight'),
                        check_dygraph=False)

    def test_check_grad_ingore_proj_weight(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'Weight', 'Bias'], ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('ProjWeight'),
                        check_dygraph=False)

    def test_check_grad_ingore_input(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Weight', 'ProjWeight', 'Bias'], ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('Input'),
                        check_dygraph=False)

    def test_check_grad_ingore_h0(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'Weight', 'ProjWeight', 'Bias', 'C0'],
                        ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('H0'),
                        check_dygraph=False)

    def test_check_grad_ingore_c0(self):
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchHidden'] = np.zeros((N, self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(['Input', 'Weight', 'ProjWeight', 'Bias', 'H0'],
                        ['Projection'],
                        numeric_grad_delta=0.0000005,
                        no_grad_set=set('C0'),
                        check_dygraph=False)


class TestLstmpOpRerverse(TestLstmpOp):

    def reset_argument(self):
        self.is_reverse = True


class TestLstmpOpNotUsePeepholes(TestLstmpOp):

    def reset_argument(self):
        self.use_peepholes = False


class TestLstmpOpLinearProjection(TestLstmpOp):

    def reset_argument(self):
        self.act_proj = 'identity'


class TestLstmpOpLen0Case1(TestLstmpOp):

    def reset_argument(self):
        self.lod = [[0, 4, 0]]


class TestLstmpOpLen0Case2(TestLstmpOp):

    def reset_argument(self):
        self.lod = [[2, 0, 3]]


class TestLstmpOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                input_data = np.random.random((1, 2048)).astype("float32")
                fluid.layers.dynamic_lstmp(input=input_data,
                                           size=2048,
                                           proj_size=256,
                                           use_peepholes=False,
                                           is_reverse=True,
                                           cell_activation="tanh",
                                           proj_activation="tanh")

            self.assertRaises(TypeError, test_Variable)

            def test_h_0():
                in_data = fluid.data(name="input",
                                     shape=[None, 2048],
                                     dtype="float32")
                h = fluid.data(name="h", shape=[None, 512], dtype="int32")
                c = fluid.data(name="c", shape=[None, 512], dtype="float32")
                fluid.layers.dynamic_lstmp(input=in_data,
                                           size=2048,
                                           proj_size=256,
                                           use_peepholes=False,
                                           is_reverse=True,
                                           cell_activation="tanh",
                                           proj_activation="tanh",
                                           h_0=h,
                                           c_0=c)

            self.assertRaises(TypeError, test_h_0)

            def test_c_0():
                in_data_ = fluid.data(name="input_",
                                      shape=[None, 2048],
                                      dtype="float32")
                h_ = fluid.data(name="h_", shape=[None, 512], dtype="float32")
                c_ = fluid.data(name="c_", shape=[None, 512], dtype="int32")
                fluid.layers.dynamic_lstmp(input=in_data_,
                                           size=2048,
                                           proj_size=256,
                                           use_peepholes=False,
                                           is_reverse=True,
                                           cell_activation="tanh",
                                           proj_activation="tanh",
                                           h_0=h_,
                                           c_0=c_)

            self.assertRaises(TypeError, test_c_0)


if __name__ == '__main__':
    unittest.main()
