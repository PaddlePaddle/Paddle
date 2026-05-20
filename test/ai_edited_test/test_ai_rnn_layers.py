# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.rnn (RNN, LSTM, GRU)
# 自动生成的单测，覆盖 paddle.nn.layer.rnn 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/nn/layer/rnn.py
# 目标：覆盖 RNN、LSTM、GRU 的各种初始化参数和前向传播路径

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. SimpleRNN - 初始化和前向传播 (各种参数组合)
2. LSTM - 初始化和前向传播 (bidirectional, layers)
3. GRU - 初始化和前向传播
4. RNNCell, LSTMCell, GRUCell - 基本功能
5. time_major 参数 / dropout 参数
"""

import unittest

import paddle
from paddle import nn


class TestSimpleRNN(unittest.TestCase):
    """Test SimpleRNN.
    测试 SimpleRNN。
    """

    def setUp(self):
        paddle.disable_static()

    def test_rnn_basic(self):
        """Basic SimpleRNN."""
        rnn = nn.SimpleRNN(input_size=16, hidden_size=32)
        x = paddle.randn([4, 10, 16])
        out, h = rnn(x)
        self.assertEqual(out.shape, [4, 10, 32])
        self.assertEqual(h.shape, [1, 4, 32])

    def test_rnn_multi_layer(self):
        """SimpleRNN with multiple layers."""
        rnn = nn.SimpleRNN(input_size=16, hidden_size=32, num_layers=3)
        x = paddle.randn([4, 10, 16])
        out, h = rnn(x)
        self.assertEqual(out.shape, [4, 10, 32])
        self.assertEqual(h.shape, [3, 4, 32])

    def test_rnn_bidirectional(self):
        """Bidirectional SimpleRNN."""
        rnn = nn.SimpleRNN(
            input_size=16, hidden_size=32, direction='bidirectional'
        )
        x = paddle.randn([4, 10, 16])
        out, h = rnn(x)
        self.assertEqual(out.shape, [4, 10, 64])
        self.assertEqual(h.shape, [2, 4, 32])

    def test_rnn_dropout(self):
        """SimpleRNN with dropout."""
        rnn = nn.SimpleRNN(
            input_size=16,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
        )
        rnn.train()
        x = paddle.randn([4, 10, 16])
        out, h = rnn(x)
        self.assertEqual(out.shape, [4, 10, 32])

    def test_rnn_time_major(self):
        """SimpleRNN with time_major=True."""
        rnn = nn.SimpleRNN(input_size=16, hidden_size=32, time_major=True)
        x = paddle.randn([10, 4, 16])
        out, h = rnn(x)
        self.assertEqual(out.shape, [10, 4, 32])

    def test_rnn_with_initial_state(self):
        """SimpleRNN with initial state."""
        rnn = nn.SimpleRNN(input_size=16, hidden_size=32)
        x = paddle.randn([4, 10, 16])
        h0 = paddle.zeros([1, 4, 32])
        out, h = rnn(x, h0)
        self.assertEqual(out.shape, [4, 10, 32])


class TestLSTM(unittest.TestCase):
    """Test LSTM.
    测试 LSTM。
    """

    def setUp(self):
        paddle.disable_static()

    def test_lstm_basic(self):
        """Basic LSTM."""
        lstm = nn.LSTM(input_size=16, hidden_size=32)
        x = paddle.randn([4, 10, 16])
        out, (h, c) = lstm(x)
        self.assertEqual(out.shape, [4, 10, 32])
        self.assertEqual(h.shape, [1, 4, 32])
        self.assertEqual(c.shape, [1, 4, 32])

    def test_lstm_multi_layer(self):
        """LSTM with multiple layers."""
        lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=3)
        x = paddle.randn([4, 10, 16])
        out, (h, c) = lstm(x)
        self.assertEqual(h.shape, [3, 4, 32])

    def test_lstm_bidirectional(self):
        """Bidirectional LSTM."""
        lstm = nn.LSTM(input_size=16, hidden_size=32, direction='bidirectional')
        x = paddle.randn([4, 10, 16])
        out, (h, c) = lstm(x)
        self.assertEqual(out.shape, [4, 10, 64])
        self.assertEqual(h.shape, [2, 4, 32])

    def test_lstm_dropout(self):
        """LSTM with dropout."""
        lstm = nn.LSTM(
            input_size=16,
            hidden_size=32,
            num_layers=3,
            dropout=0.1,
        )
        lstm.train()
        x = paddle.randn([4, 10, 16])
        out, _ = lstm(x)
        self.assertEqual(out.shape, [4, 10, 32])

    def test_lstm_time_major(self):
        """LSTM with time_major=True."""
        lstm = nn.LSTM(input_size=16, hidden_size=32, time_major=True)
        x = paddle.randn([10, 4, 16])
        out, _ = lstm(x)
        self.assertEqual(out.shape, [10, 4, 32])


class TestGRU(unittest.TestCase):
    """Test GRU.
    测试 GRU。
    """

    def setUp(self):
        paddle.disable_static()

    def test_gru_basic(self):
        """Basic GRU."""
        gru = nn.GRU(input_size=16, hidden_size=32)
        x = paddle.randn([4, 10, 16])
        out, h = gru(x)
        self.assertEqual(out.shape, [4, 10, 32])

    def test_gru_bidirectional(self):
        """Bidirectional GRU."""
        gru = nn.GRU(input_size=16, hidden_size=32, direction='bidirectional')
        x = paddle.randn([4, 10, 16])
        out, h = gru(x)
        self.assertEqual(out.shape, [4, 10, 64])

    def test_gru_multi_layer(self):
        """GRU with multiple layers."""
        gru = nn.GRU(input_size=16, hidden_size=32, num_layers=3)
        x = paddle.randn([4, 10, 16])
        out, h = gru(x)
        self.assertEqual(h.shape, [3, 4, 32])


class TestRNNCells(unittest.TestCase):
    """Test RNN cells.
    测试 RNN 单元。
    """

    def setUp(self):
        paddle.disable_static()

    def test_rnn_cell(self):
        """SimpleRNNCell - returns output and hidden state."""
        cell = nn.SimpleRNNCell(input_size=16, hidden_size=32)
        x = paddle.randn([4, 16])
        h = paddle.zeros([4, 32])
        out, _ = cell(x, h)
        self.assertEqual(out.shape, [4, 32])

    def test_lstm_cell(self):
        """LSTMCell - states is a list/tuple, forward returns (output, (h, c))."""
        cell = nn.LSTMCell(input_size=16, hidden_size=32)
        x = paddle.randn([4, 16])
        h = paddle.zeros([4, 32])
        c = paddle.zeros([4, 32])
        out, (h_new, c_new) = cell(x, states=[h, c])
        self.assertEqual(h_new.shape, [4, 32])
        self.assertEqual(c_new.shape, [4, 32])

    def test_gru_cell(self):
        """GRUCell - forward returns (h_new, new_states)."""
        cell = nn.GRUCell(input_size=16, hidden_size=32)
        x = paddle.randn([4, 16])
        h = paddle.zeros([4, 32])
        h_new, _ = cell(x, h)
        self.assertEqual(h_new.shape, [4, 32])

    def test_rnn_cell_sequence(self):
        """Process sequence with SimpleRNNCell."""
        cell = nn.SimpleRNNCell(input_size=16, hidden_size=32)
        x = paddle.randn([10, 4, 16])
        h = paddle.zeros([4, 32])
        for t in range(10):
            h, _ = cell(x[t], h)
        self.assertEqual(h.shape, [4, 32])


if __name__ == '__main__':
    unittest.main()
