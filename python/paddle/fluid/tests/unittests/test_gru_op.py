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

import unittest
import numpy as np
import math
from op_test import OpTest
from test_lstm_op import identity, sigmoid, tanh, relu


class TestGRUOp(OpTest):
    lod = [[2, 4, 3]]
    batch_size = sum(lod[0])
    frame_size = 5
    activate = {
        'identity': identity,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu
    }

    @staticmethod
    def seq_to_batch(lod, is_reverse):
        idx_in_seq_list = []
        seq_lens = lod[0]
        seq_starts = [0]
        for i in range(len(seq_lens)):
            seq_starts.append(seq_starts[-1] + seq_lens[i])
        sorted_seqs = sorted(
            range(len(seq_lens)), lambda x, y: seq_lens[y] - seq_lens[x])
        num_batch = seq_lens[sorted_seqs[0]]
        for batch_idx in range(num_batch):
            idx_in_seq = []
            for i in range(len(seq_lens)):
                if seq_lens[sorted_seqs[i]] <= batch_idx:
                    break
                idx = (seq_starts[sorted_seqs[i] + 1] - 1 - batch_idx
                       ) if is_reverse else (
                           seq_starts[sorted_seqs[i]] + batch_idx)
                idx_in_seq.append(idx)
            idx_in_seq_list.append(idx_in_seq)
        return idx_in_seq_list, sorted_seqs

    def gru_step(self, x, h_p, w, b):
        batch_size = x.shape[0]
        frame_size = w.shape[0]
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
        return g, r_h_p, h

    def gru(self):
        input, lod = self.inputs['Input']
        w = self.inputs['Weight']
        b = self.inputs['Bias'] if self.inputs.has_key('Bias') else np.zeros(
            (1, self.frame_size * 3))
        batch_gate = self.outputs['BatchGate']
        batch_reset_hidden_prev = self.outputs['BatchResetHiddenPrev']
        batch_hidden = self.outputs['BatchHidden']
        hidden = self.outputs['Hidden']
        idx_in_seq_list = self.idx_in_seq_list
        h_p = self.inputs['H0'][self.sorted_seqs] if self.inputs.has_key(
            'H0') else np.zeros((len(idx_in_seq_list[0]), self.frame_size))
        num_batch = len(idx_in_seq_list)
        end_idx = 0
        for batch_idx in range(num_batch):
            x = input[idx_in_seq_list[batch_idx]]
            g, r_h_p, h = self.gru_step(x, h_p, w, b)
            if batch_idx < (num_batch - 1):
                h_p = h[:len(idx_in_seq_list[batch_idx + 1])]
            start_idx = end_idx
            end_idx = start_idx + len(idx_in_seq_list[batch_idx])
            batch_gate[start_idx:end_idx] = g
            batch_reset_hidden_prev[start_idx:end_idx] = r_h_p
            batch_hidden[start_idx:end_idx] = h
            hidden[idx_in_seq_list[batch_idx]] = h
        return batch_gate, batch_reset_hidden_prev, hidden

    def set_data(self):
        lod = self.lod
        self.idx_in_seq_list, self.sorted_seqs = self.seq_to_batch(
            lod, self.is_reverse)
        batch_size = self.batch_size
        frame_size = self.frame_size
        input = np.random.rand(batch_size, frame_size * 3).astype('float64')
        h0 = np.random.rand(len(self.idx_in_seq_list[0]),
                            frame_size).astype('float64')
        weight = np.random.rand(frame_size, frame_size * 3).astype('float64')
        bias = np.random.rand(1, frame_size * 3).astype('float64')

        self.inputs = {
            'Input': (input, lod),
            'H0': h0,
            'Weight': weight,
            'Bias': bias
        }

        self.outputs = {
            'BatchGate': np.zeros(
                (batch_size, frame_size * 3), dtype='float64'),
            'BatchResetHiddenPrev': np.zeros(
                (batch_size, frame_size), dtype='float64'),
            'BatchHidden': np.zeros(
                (batch_size, frame_size), dtype='float64'),
            'Hidden': np.zeros(
                (batch_size, frame_size), dtype='float64')
        }

    def set_confs(self):
        self.is_reverse = False
        self.attrs = {
            'activation': 'tanh',
            'gate_activation': 'sigmoid',
            'is_reverse': self.is_reverse
        }

    def setUp(self):
        self.op_type = "gru"
        self.set_confs()
        self.set_data()
        self.gru()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input', 'H0', 'Weight', 'Bias'], ['Hidden'])


class TestGRUOpNoInitial(TestGRUOp):
    def set_data(self):
        super(TestGRUOpNoInitial, self).set_data()
        self.inputs.pop('H0')

    def test_check_grad(self):
        self.check_grad(['Input', 'Weight', 'Bias'], ['Hidden'])


class TestGRUOpReverse(TestGRUOp):
    def set_confs(self):
        self.is_reverse = True
        self.attrs = {
            'activation': 'tanh',
            'gate_activation': 'sigmoid',
            'is_reverse': self.is_reverse
        }


if __name__ == "__main__":
    unittest.main()
