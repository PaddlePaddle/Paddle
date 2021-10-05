# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
paddle.framework.set_default_dtype("float64")

import numpy as np
import unittest

from rnn_numpy import SimpleRNNCell, LSTMCell, GRUCell, BiRNN
from convert import convert_params_for_cell


class TestSimpleRNNCell(unittest.TestCase):
    def __init__(self, bias=True, place="cpu"):
        super(TestSimpleRNNCell, self).__init__(methodName="runTest")
        self.bias = bias
        self.place = paddle.CPUPlace() if place == "cpu" \
            else paddle.CUDAPlace(0)

    def setUp(self):
        paddle.disable_static(self.place)
        rnn1 = SimpleRNNCell(16, 32, bias=self.bias)
        rnn2 = paddle.nn.SimpleRNNCell(
            16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)

        y1, h1 = rnn1(x, prev_h)
        y2, h2 = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)

        y1, h1 = rnn1(x)
        y2, h2 = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_errors(self):
        def test_zero_hidden_size():
            cell = paddle.nn.SimpleRNNCell(-1, 0)

        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()


class TestGRUCell(unittest.TestCase):
    def __init__(self, bias=True, place="cpu"):
        super(TestGRUCell, self).__init__(methodName="runTest")
        self.bias = bias
        self.place = paddle.CPUPlace() if place == "cpu" \
            else paddle.CUDAPlace(0)

    def setUp(self):
        paddle.disable_static(self.place)
        rnn1 = GRUCell(16, 32, bias=self.bias)
        rnn2 = paddle.nn.GRUCell(
            16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)

        y1, h1 = rnn1(x, prev_h)
        y2, h2 = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)

        y1, h1 = rnn1(x)
        y2, h2 = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_errors(self):
        def test_zero_hidden_size():
            cell = paddle.nn.GRUCell(-1, 0)

        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()


class TestLSTMCell(unittest.TestCase):
    def __init__(self, bias=True, place="cpu"):
        super(TestLSTMCell, self).__init__(methodName="runTest")
        self.bias = bias
        self.place = paddle.CPUPlace() if place == "cpu" \
            else paddle.CUDAPlace(0)

    def setUp(self):
        rnn1 = LSTMCell(16, 32, bias=self.bias)
        rnn2 = paddle.nn.LSTMCell(
            16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        convert_params_for_cell(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)
        prev_h = np.random.randn(4, 32)
        prev_c = np.random.randn(4, 32)

        y1, (h1, c1) = rnn1(x, (prev_h, prev_c))
        y2, (h2, c2) = rnn2(
            paddle.to_tensor(x),
            (paddle.to_tensor(prev_h), paddle.to_tensor(prev_c)))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(4, 16)

        y1, (h1, c1) = rnn1(x)
        y2, (h2, c2) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-8, rtol=1e-5)

    def test_errors(self):
        def test_zero_hidden_size():
            cell = paddle.nn.LSTMCell(-1, 0)

        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()


class TestBiRNN(unittest.TestCase):
    def __init__(self, bias=True, place="cpu", time_major=False) -> None:
        super(TestBiRNN, self).__init__(methodName="runTest")
        self.bias = bias
        self.place = paddle.CUDAPlace(0) if place == "gpu" \
                else paddle.CPUPlace()
        self.time_major = time_major

    def setUp(self):
        cell_fw1 = SimpleRNNCell(16, 32, bias=self.bias)
        cell_bw1 = SimpleRNNCell(16, 32, bias=self.bias)
        cell_fw2 = paddle.nn.SimpleRNNCell(
            16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
        cell_bw2 = paddle.nn.SimpleRNNCell(
            16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)

        convert_params_for_cell(cell_fw1, cell_fw2)
        convert_params_for_cell(cell_bw1, cell_bw2)

        rnn1 = BiRNN(cell_fw1, cell_bw1, time_major=self.time_major)
        rnn2 = paddle.nn.BiRNN(cell_fw2, cell_bw2, time_major=self.time_major)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        bz = 3
        time_step = 10
        size = 16
        if self.time_major:
            x = np.random.randn(time_step, bz, size)
        else:
            x = np.random.randn(bz, time_step, size)

        fw_states = np.random.randn(bz, size * 2)
        bw_states = np.random.randn(bz, size * 2)
        initial_states = (fw_states, bw_states)
        y1, f1 = rnn1(x, initial_states)
        y2, f2 = rnn2(
            paddle.to_tensor(x),
            paddle.to_tensor(
                initial_states, place=self.place))

        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)

        for it in zip(f1, f2):
            np.testing.assert_allclose(
                it[0], it[1].numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        bz = 3
        time_step = 10
        size = 16
        if self.time_major:
            x = np.random.randn(time_step, bz, size)
        else:
            x = np.random.randn(bz, time_step, size)

        y1, f1 = rnn1(x)
        y2, f2 = rnn2(paddle.to_tensor(x, place=self.place))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)

        for it in zip(f1, f2):
            np.testing.assert_allclose(
                it[0], it[1].numpy(), atol=1e-8, rtol=1e-5)

    def test_errors(self):
        def test_zero_hidden_size():
            cell_fw = paddle.nn.SimpleRNNCell(
                10, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
            cell_bw = paddle.nn.SimpleRNNCell(
                16, 32, bias_ih_attr=self.bias, bias_hh_attr=self.bias)
            paddle.nn.BiRNN(cell_fw, cell_bw)

        self.assertRaises(ValueError, test_zero_hidden_size)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_errors()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    devices = ["cpu", "gpu"] if paddle.fluid.is_compiled_with_cuda() \
        else ["cpu"]

    for bias in [True, False]:
        for device in devices:
            for test_class in [
                    TestLSTMCell, TestSimpleRNNCell, TestGRUCell, TestLSTMCell
            ]:
                suite.addTest(test_class(bias, device))

    for bias in [True, False]:
        for device in devices:
            for time_major in [True, False]:
                for test_class in [TestBiRNN]:
                    suite.addTest(test_class(bias, device, time_major))
    return suite


if __name__ == '__main__':
    unittest.main()
