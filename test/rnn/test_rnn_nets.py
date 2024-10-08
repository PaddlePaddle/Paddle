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
import os
import sys
import tempfile
import unittest

import numpy as np
from convert import convert_params_for_net

sys.path.append("../../rnn")
from rnn_numpy import GRU, LSTM, SimpleRNN

bidirectional_list = ["bidirectional", "bidirect"]


class TestSimpleRNN(unittest.TestCase):
    def __init__(
        self, time_major=True, direction="forward", place="cpu", mode='RNN_TANH'
    ):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place
        self.mode = mode

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        paddle.disable_static(place)
        rnn1 = SimpleRNN(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            nonlinearity=self.mode,
        )
        rnn2 = paddle.nn.SimpleRNN(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            activation=self.mode[4:].lower(),
        )
        convert_params_for_net(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)
        y2, h2 = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)
        y2, h2 = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_predict(self):
        predict_test_util(self.place, "SimpleRNN")

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_predict()


class TestGRU(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        paddle.disable_static(place)
        rnn1 = GRU(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )
        rnn2 = paddle.nn.GRU(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )
        convert_params_for_net(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(2 * self.num_directions, 4, 32)

        y1, h1 = rnn1(x, prev_h)
        y2, h2 = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)
        y2, h2 = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_predict(self):
        predict_test_util(self.place, "GRU")

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_predict()


class TestLSTM(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        paddle.disable_static(place)
        rnn1 = LSTM(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )
        rnn2 = paddle.nn.LSTM(
            16, 32, 2, time_major=self.time_major, direction=self.direction
        )
        convert_params_for_net(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(
            2 * self.num_directions, 4, getattr(self, "proj_size", 32)
        )
        prev_c = np.random.randn(2 * self.num_directions, 4, 32)

        y1, (h1, c1) = rnn1(x, (prev_h, prev_c))
        y2, (h2, c2) = rnn2(
            paddle.to_tensor(x),
            (paddle.to_tensor(prev_h), paddle.to_tensor(prev_c)),
        )
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, (h1, c1) = rnn1(x)
        y2, (h2, c2) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(c1, c2.numpy(), atol=1e-8, rtol=1e-5)

    def test_predict(self):
        predict_test_util(self.place, "LSTM")
        predict_test_util(self.place, "LSTM", False)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_predict()


class TestLSTMWithProjSize(TestLSTM):
    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.
        place = paddle.set_device(self.place)
        paddle.disable_static(place)
        rnn1 = LSTM(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            proj_size=8,
        )
        rnn2 = paddle.nn.LSTM(
            16,
            32,
            2,
            time_major=self.time_major,
            direction=self.direction,
            proj_size=8,
        )
        convert_params_for_net(rnn1, rnn2)

        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.proj_size = 8


def predict_test_util(place, mode, stop_gradient=True):
    place = paddle.set_device(place)
    paddle.seed(123)
    np.random.seed(123)

    class Net(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.rnn = getattr(paddle.nn, mode)(
                16, 32, 2, direction="bidirectional", dropout=0.1
            )

        def forward(self, input):
            return self.rnn(input)

    x = paddle.randn((4, 10, 16))
    x.stop_gradient = stop_gradient
    seq_len = paddle.to_tensor(np.array([10, 6, 8, 5]))
    mask = paddle.static.nn.sequence_lod.sequence_mask(
        seq_len, maxlen=10, dtype=x.dtype
    )
    mask = paddle.unsqueeze(mask, [2])
    rnn = Net()
    y, _ = rnn(x)
    y = y * mask
    loss = paddle.mean(y)
    loss.backward()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.1, parameters=rnn.parameters()
    )
    optimizer.step()
    rnn.eval()
    y, _ = rnn(x)
    # `jit.to_static` would include a train_program, eval mode might cause
    # some errors currently, such as dropout grad op gets `is_test == True`.
    rnn.train()

    rnn = paddle.jit.to_static(
        rnn,
        [paddle.static.InputSpec(shape=[None, None, 16], dtype=x.dtype)],
        full_graph=True,
    )
    temp_dir = tempfile.TemporaryDirectory()
    save_dirname = os.path.join(temp_dir.name, f"./inference/{mode}_infer")

    paddle.jit.save(rnn, save_dirname)

    paddle.enable_static()

    new_scope = paddle.static.Scope()
    with paddle.static.scope_guard(new_scope):
        exe = paddle.static.Executor(place)
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(save_dirname, exe)
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: x.numpy()},
            fetch_list=fetch_targets,
        )
        np.testing.assert_equal(
            y.numpy(), results[0]
        )  # eval results equal predict results
    paddle.disable_static()

    temp_dir.cleanup()


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


if __name__ == '__main__':
    unittest.main()
