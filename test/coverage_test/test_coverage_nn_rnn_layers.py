# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
RNN层单元测试 / RNN Layers Unit Tests

测试目标 / Test Target:
  paddle.nn RNN相关层

覆盖的模块 / Covered Modules:
  - paddle.nn.SimpleRNN: 简单RNN
  - paddle.nn.LSTM: 长短期记忆网络
  - paddle.nn.GRU: 门控循环单元
  - paddle.nn.RNNCell: RNN单元
  - paddle.nn.LSTMCell: LSTM单元
  - paddle.nn.GRUCell: GRU单元

作用 / Purpose:
  补充RNN层API的各种参数组合测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestSimpleRNN(unittest.TestCase):
    """测试SimpleRNN / Test SimpleRNN"""

    def test_rnn_basic(self):
        """测试基本RNN / Test basic RNN"""
        rnn = nn.SimpleRNN(16, 32, num_layers=1)
        x = paddle.randn([4, 10, 16])  # [batch, seq_len, input_size]
        output, hidden = rnn(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(hidden.shape, [1, 4, 32])

    def test_rnn_multilayer(self):
        """测试多层RNN / Test multi-layer RNN"""
        rnn = nn.SimpleRNN(16, 32, num_layers=2)
        x = paddle.randn([4, 10, 16])
        output, hidden = rnn(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(hidden.shape, [2, 4, 32])

    def test_rnn_bidirectional(self):
        """测试双向RNN / Test bidirectional RNN"""
        rnn = nn.SimpleRNN(16, 32, direction='bidirect')
        x = paddle.randn([4, 10, 16])
        output, hidden = rnn(x)
        self.assertEqual(output.shape, [4, 10, 64])  # 32*2 for bidirectional

    def test_rnn_time_major(self):
        """测试时间优先RNN / Test time_major RNN"""
        rnn = nn.SimpleRNN(16, 32, time_major=True)
        x = paddle.randn([10, 4, 16])  # [seq_len, batch, input_size]
        output, hidden = rnn(x)
        self.assertEqual(output.shape, [10, 4, 32])


class TestLSTM(unittest.TestCase):
    """测试LSTM / Test LSTM"""

    def test_lstm_basic(self):
        """测试基本LSTM / Test basic LSTM"""
        lstm = nn.LSTM(16, 32)
        x = paddle.randn([4, 10, 16])
        output, (h, c) = lstm(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(h.shape, [1, 4, 32])
        self.assertEqual(c.shape, [1, 4, 32])

    def test_lstm_multilayer(self):
        """测试多层LSTM / Test multi-layer LSTM"""
        lstm = nn.LSTM(16, 32, num_layers=2)
        x = paddle.randn([4, 10, 16])
        output, (h, c) = lstm(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(h.shape, [2, 4, 32])

    def test_lstm_bidirectional(self):
        """测试双向LSTM / Test bidirectional LSTM"""
        lstm = nn.LSTM(16, 32, direction='bidirect')
        x = paddle.randn([4, 10, 16])
        output, (h, c) = lstm(x)
        self.assertEqual(output.shape, [4, 10, 64])

    def test_lstm_with_dropout(self):
        """测试带dropout的LSTM / Test LSTM with dropout"""
        lstm = nn.LSTM(16, 32, num_layers=2, dropout=0.5)
        lstm.train()
        x = paddle.randn([4, 10, 16])
        output, (h, c) = lstm(x)
        self.assertEqual(output.shape, [4, 10, 32])

    def test_lstm_initial_states(self):
        """测试带初始状态的LSTM / Test LSTM with initial states"""
        lstm = nn.LSTM(16, 32)
        x = paddle.randn([4, 10, 16])
        h0 = paddle.zeros([1, 4, 32])
        c0 = paddle.zeros([1, 4, 32])
        output, (h, c) = lstm(x, initial_states=(h0, c0))
        self.assertEqual(output.shape, [4, 10, 32])


class TestGRU(unittest.TestCase):
    """测试GRU / Test GRU"""

    def test_gru_basic(self):
        """测试基本GRU / Test basic GRU"""
        gru = nn.GRU(16, 32)
        x = paddle.randn([4, 10, 16])
        output, hidden = gru(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(hidden.shape, [1, 4, 32])

    def test_gru_multilayer(self):
        """测试多层GRU / Test multi-layer GRU"""
        gru = nn.GRU(16, 32, num_layers=2)
        x = paddle.randn([4, 10, 16])
        output, hidden = gru(x)
        self.assertEqual(output.shape, [4, 10, 32])
        self.assertEqual(hidden.shape, [2, 4, 32])

    def test_gru_bidirectional(self):
        """测试双向GRU / Test bidirectional GRU"""
        gru = nn.GRU(16, 32, direction='bidirect')
        x = paddle.randn([4, 10, 16])
        output, hidden = gru(x)
        self.assertEqual(output.shape, [4, 10, 64])


class TestRNNCells(unittest.TestCase):
    """测试RNN单元 / Test RNN Cells"""

    def test_rnn_cell(self):
        """测试RNNCell / Test RNNCell"""
        cell = nn.SimpleRNNCell(16, 32)
        x = paddle.randn([4, 16])
        prev_h = paddle.zeros([4, 32])
        y, h = cell(x, prev_h)
        self.assertEqual(y.shape, [4, 32])

    def test_lstm_cell(self):
        """测试LSTMCell / Test LSTMCell"""
        cell = nn.LSTMCell(16, 32)
        x = paddle.randn([4, 16])
        prev_h = paddle.zeros([4, 32])
        prev_c = paddle.zeros([4, 32])
        y, (h, c) = cell(x, (prev_h, prev_c))
        self.assertEqual(y.shape, [4, 32])
        self.assertEqual(c.shape, [4, 32])

    def test_gru_cell(self):
        """测试GRUCell / Test GRUCell"""
        cell = nn.GRUCell(16, 32)
        x = paddle.randn([4, 16])
        prev_h = paddle.zeros([4, 32])
        y, h = cell(x, prev_h)
        self.assertEqual(y.shape, [4, 32])


if __name__ == '__main__':
    unittest.main()
