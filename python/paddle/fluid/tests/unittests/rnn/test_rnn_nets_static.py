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
paddle.set_default_dtype("float64")
from paddle.fluid.layers import sequence_mask
paddle.enable_static()

import numpy as np
import unittest

from convert import convert_params_for_net_static
from rnn_numpy import SimpleRNN, LSTM, GRU

bidirectional_list = ["bidirectional", "bidirect"]


class TestSimpleRNN(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super(TestSimpleRNN, self).__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = SimpleRNN(
            16, 32, 2, time_major=self.time_major, direction=self.direction)

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.SimpleRNN(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction)

        exe = paddle.static.Executor(place)
        scope = paddle.fluid.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2

        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        mp = self.mp.clone().clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                init_h = paddle.fluid.data(
                    "init_h", [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype())
                y, h = rnn2(x_data, init_h)

        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                y, h = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_input_lengths(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)

        y1, h1 = rnn1(x, sequence_length=sequence_length)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.fluid.data("seq_len", [-1], dtype="int64")
                mask = sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                y, h = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)

        feed_dict = {x_data.name: x, seq_len.name: sequence_length}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()


class TestGRU(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super(TestGRU, self).__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = GRU(16,
                   32,
                   2,
                   time_major=self.time_major,
                   direction=self.direction)

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.GRU(16,
                                     32,
                                     2,
                                     time_major=self.time_major,
                                     direction=self.direction)

        exe = paddle.static.Executor(place)
        scope = paddle.fluid.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2

        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                init_h = paddle.fluid.data(
                    "init_h", [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype())
                y, h = rnn2(x_data, init_h)

        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                y, h = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_input_lengths(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)

        y1, h1 = rnn1(x, sequence_length=sequence_length)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.fluid.data("seq_len", [-1], dtype="int64")
                mask = sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                y, h = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)

        feed_dict = {x_data.name: x, seq_len.name: sequence_length}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()


class TestLSTM(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super(TestLSTM, self).__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = LSTM(
            16, 32, 2, time_major=self.time_major, direction=self.direction)

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction)

        exe = paddle.static.Executor(place)
        scope = paddle.fluid.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        self.mp = mp
        self.sp = sp
        self.rnn1 = rnn1
        self.rnn2 = rnn2

        self.place = place
        self.executor = exe
        self.scope = scope

    def test_with_initial_state(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)
        prev_c = np.random.randn(2 * self.num_directions, 4, 32)

        y1, (h1, c1) = rnn1(x, (prev_h, prev_c))

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                init_h = paddle.fluid.data(
                    "init_h", [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype())
                init_c = paddle.fluid.data(
                    "init_c", [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype())
                y, (h, c) = rnn2(x_data, (init_h, init_c))

        feed_dict = {x_data.name: x, init_h.name: prev_h, init_c.name: prev_c}
        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, (h1, c1) = rnn1(x)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                y, (h, c) = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def test_with_input_lengths(self):
        mp = self.mp.clone()
        sp = self.sp
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        exe = self.executor
        scope = self.scope

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)

        y1, (h1, c1) = rnn1(x, sequence_length=sequence_length)

        with paddle.fluid.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.fluid.data(
                    "input", [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype())
                seq_len = paddle.fluid.data("seq_len", [-1], dtype="int64")
                mask = sequence_mask(seq_len, dtype=paddle.get_default_dtype())
                if self.time_major:
                    mask = paddle.transpose(mask, [1, 0])
                y, (h, c) = rnn2(x_data, sequence_length=seq_len)
                mask = paddle.unsqueeze(mask, -1)
                y = paddle.multiply(y, mask)

        feed_dict = {x_data.name: x, seq_len.name: sequence_length}

        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    devices = ["cpu", "gpu"] if paddle.fluid.is_compiled_with_cuda() \
        else ["cpu"]
    for direction in ["forward", "bidirectional", "bidirect"]:
        for time_major in [True, False]:
            for device in devices:
                for test_class in [TestSimpleRNN, TestLSTM, TestGRU]:
                    suite.addTest(test_class(time_major, direction, device))
    return suite


if __name__ == "__main__":
    unittest.main()
