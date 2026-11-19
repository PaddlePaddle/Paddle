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

# Correctness tests for the custom-device PIR interpreter Run boundary.
#
# Before running a Run's instruction list the interpreter orders side streams
# against the default (eager-shared) stream: a default-stream-only Run does
# nothing (stream FIFO orders the eager handoff); a Run that uses side streams
# records an event on the default stream and makes each side stream wait on it
# asynchronously, so side-stream ops run after all prior default-stream work.
#
# These tests build the custom_cpu plugin (via PLUGIN_URL / PLUGIN_TAG, as the
# sibling custom_runtime tests do) so they actually exercise the custom-device
# path on CI, and verify the compiled result matches an eager reference.

import os
import sys
import tempfile
import unittest

import numpy as np


def _build_custom_cpu(temp_dir):
    cmd = 'cd {} \
        && git clone --depth 1 {} \
        && cd PaddleCustomDevice \
        && git fetch origin \
        && git checkout {} -b dev \
        && cd backends/custom_cpu \
        && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} \
           -DWITH_TESTING=OFF && make -j8'.format(
        temp_dir,
        os.getenv('PLUGIN_URL'),
        os.getenv('PLUGIN_TAG'),
        sys.executable,
    )
    os.system(cmd)
    os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
        temp_dir, 'PaddleCustomDevice/backends/custom_cpu/build'
    )


class TestCustomDeviceRunBoundary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _build_custom_cpu(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        os.environ.pop('CUSTOM_DEVICE_ROOT', None)

    def _net(self, paddle):
        class Net(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = paddle.nn.Linear(64, 128)
                self.fc2 = paddle.nn.Linear(128, 64)

            def forward(self, x):
                h = paddle.nn.functional.relu(self.fc1(x))
                return paddle.nn.functional.relu(self.fc2(h))

        return Net()

    def _train(self, paddle, net, x, steps):
        opt = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=net.parameters()
        )
        losses = []
        for _ in range(steps):
            loss = net(x).mean()
            loss.backward()
            opt.step()
            opt.clear_grad()
            losses.append(float(loss))
        return np.array(losses)

    def test_compiled_matches_eager(self):
        import paddle

        if "custom_cpu" not in (
            paddle.device.get_all_custom_device_type() or []
        ):
            self.skipTest("custom_cpu plugin not available")

        x_np = np.random.RandomState(0).randn(4, 64).astype("float32")

        paddle.set_device("custom_cpu")
        paddle.seed(2026)
        loss_eager = self._train(
            paddle, self._net(paddle), paddle.to_tensor(x_np), 50
        )

        paddle.seed(2026)
        net = paddle.jit.to_static(
            self._net(paddle),
            full_graph=True,
            input_spec=[paddle.static.InputSpec([4, 64], "float32")],
        )
        loss_static = self._train(paddle, net, paddle.to_tensor(x_np), 50)

        np.testing.assert_allclose(
            loss_static, loss_eager, rtol=1e-5, atol=1e-5
        )

    def test_cpu_readback_no_race(self):
        import paddle

        if "custom_cpu" not in (
            paddle.device.get_all_custom_device_type() or []
        ):
            self.skipTest("custom_cpu plugin not available")
        paddle.set_device("custom_cpu")

        class CpuNet(paddle.nn.Layer):
            def forward(self, x):
                return (paddle.nn.functional.relu(x * 2.0 - 1.0) + 3.0).cpu()

        n = 1 << 16
        net = paddle.jit.to_static(
            CpuNet(),
            full_graph=True,
            input_spec=[paddle.static.InputSpec([n], "float32")],
        )
        base = np.random.RandomState(0).randn(n).astype("float32")
        for i in range(50):
            out = net(paddle.to_tensor(base + i)).numpy()
            expect = np.maximum((base + i) * 2.0 - 1.0, 0.0) + 3.0
            np.testing.assert_allclose(out, expect, rtol=0, atol=0)


if __name__ == "__main__":
    if os.name == "nt" or sys.platform.startswith("darwin"):
        sys.exit()
    unittest.main()
