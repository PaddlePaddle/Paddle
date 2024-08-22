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


paddle.enable_static()

import sys
import unittest

import numpy as np
from convert import convert_params_for_net_static

sys.path.append("../../rnn")
from rnn_numpy import GRU, LSTM, SimpleRNN

bidirectional_list = ["bidirectional", "bidirect"]


class TestSimpleRNN(unittest.TestCase):
    def __init__(
        self, time_major=True, direction="forward", place="cpu", mode="RNN_TANH"
    ):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place
        self.mode = mode

    def test_with_initial_state(self):
        place = paddle.set_device(self.place)
        rnn1 = SimpleRNN(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            nonlinearity=self.mode,
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.SimpleRNN(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                    activation=self.mode[4:].lower(),
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_h = paddle.static.data(
                    "init_h",
                    [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, h = rnn2(x_data, init_h)

        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        place = paddle.set_device(self.place)
        rnn1 = SimpleRNN(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            nonlinearity=self.mode,
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.SimpleRNN(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                    activation=self.mode[4:].lower(),
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, h = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()


class TestGRU(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def test_with_initial_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = GRU(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.GRU(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_h = paddle.static.data(
                    "init_h",
                    [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, h = rnn2(x_data, init_h)

        feed_dict = {x_data.name: x, init_h.name: prev_h}
        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = GRU(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.GRU(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, h = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()


class TestLSTM(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def test_with_initial_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = LSTM(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(
            2 * self.num_directions, 4, getattr(self, "proj_size", 32)
        )
        prev_c = np.random.randn(2 * self.num_directions, 4, 32)

        y1, (h1, c1) = rnn1(x, (prev_h, prev_c))

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_h = paddle.static.data(
                    "init_h",
                    [
                        2 * self.num_directions,
                        -1,
                        getattr(self, "proj_size", 32),
                    ],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_c = paddle.static.data(
                    "init_c",
                    [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, (h, c) = rnn2(x_data, (init_h, init_c))

        feed_dict = {x_data.name: x, init_h.name: prev_h, init_c.name: prev_c}
        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = LSTM(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, (h1, c1) = rnn1(x)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, (h, c) = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()


class TestLSTMWithProjSize(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def test_with_initial_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = LSTM(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            proj_size=8,
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                    proj_size=8,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)
        self.proj_size = 8

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(
            2 * self.num_directions, 4, getattr(self, "proj_size", 32)
        )
        prev_c = np.random.randn(2 * self.num_directions, 4, 32)

        y1, (h1, c1) = rnn1(x, (prev_h, prev_c))

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_h = paddle.static.data(
                    "init_h",
                    [
                        2 * self.num_directions,
                        -1,
                        getattr(self, "proj_size", 32),
                    ],
                    dtype=paddle.framework.get_default_dtype(),
                )
                init_c = paddle.static.data(
                    "init_c",
                    [2 * self.num_directions, -1, 32],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, (h, c) = rnn2(x_data, (init_h, init_c))

        feed_dict = {x_data.name: x, init_h.name: prev_h, init_c.name: prev_c}
        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        rnn1 = LSTM(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            proj_size=8,
        )

        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                rnn2 = paddle.nn.LSTM(
                    16,
                    32,
                    2,
                    time_major=self.time_major,
                    direction=self.direction,
                    proj_size=8,
                )

        exe = paddle.static.Executor(place)
        scope = paddle.base.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(sp)
            convert_params_for_net_static(rnn1, rnn2, place)
        self.proj_size = 8

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, (h1, c1) = rnn1(x)

        with paddle.base.unique_name.guard():
            with paddle.static.program_guard(mp, sp):
                x_data = paddle.static.data(
                    "input",
                    [-1, -1, 16],
                    dtype=paddle.framework.get_default_dtype(),
                )
                y, (h, c) = rnn2(x_data)

        feed_dict = {x_data.name: x}

        with paddle.static.scope_guard(scope):
            y2, h2, c2 = exe.run(mp, feed=feed_dict, fetch_list=[y, h, c])

        np.testing.assert_allclose(y1, y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    devices = ["cpu", "gpu"] if paddle.base.is_compiled_with_cuda() else ["cpu"]
    for direction in ["forward", "bidirectional", "bidirect"]:
        for time_major in [True, False]:
            for device in devices:
                for test_class in [
                    TestSimpleRNN,
                    TestLSTM,
                    TestGRU,
                    TestLSTMWithProjSize,
                ]:
                    suite.addTest(test_class(time_major, direction, device))
                    if test_class == TestSimpleRNN:
                        suite.addTest(
                            test_class(
                                time_major, direction, device, mode="RNN_RELU"
                            )
                        )
    return suite


if __name__ == "__main__":
    unittest.main()
