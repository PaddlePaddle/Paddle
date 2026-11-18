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

# Correctness tests for the custom-device PIR interpreter Run boundary, which
# skips the per-Run blocking default-stream sync when a Run used only the
# default (eager-shared) stream.
#
# Known limitations (see PR description; to be addressed on other backends):
#   * Cross-Run named/side-stream consumer: if a later Run's first op reads a
#     previous default-stream Run's output on a user execution stream, there is
#     no cross-Run event ordering on custom devices (the jit-input event is
#     built only for is_gpu_place), so the fast path may race. Not covered here.
#   * Host-read via an async D2H kernel (e.g. a backend that overrides memcpy
#     with blocking=false) can read before the copy completes. The generic
#     .cpu() path uses a blocking copy and is covered below.

import unittest

import numpy as np

import paddle


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(64, 128)
        self.fc2 = paddle.nn.Linear(128, 64)

    def forward(self, x):
        h = paddle.nn.functional.relu(self.fc1(x))
        return paddle.nn.functional.relu(self.fc2(h))


class CpuNet(paddle.nn.Layer):
    def forward(self, x):
        y = paddle.nn.functional.relu(x * 2.0 - 1.0) + 3.0
        return y.cpu()


def _train(net, x, steps):
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=net.parameters())
    losses = []
    for _ in range(steps):
        loss = net(x).mean()
        loss.backward()
        opt.step()
        opt.clear_grad()
        losses.append(float(loss))
    return np.array(losses)


class TestCustomDeviceRunBoundary(unittest.TestCase):
    def setUp(self):
        self.devs = paddle.device.get_all_custom_device_type() or []

    def test_compiled_matches_eager(self):
        # The interpreter shares the default stream with eager; skipping the
        # per-Run sync on a default-stream-only Run must not corrupt training.
        if not self.devs:
            self.skipTest("no custom device available")
        dev = f"{self.devs[0]}:0"

        paddle.set_device(dev)
        paddle.seed(2026)
        x = paddle.to_tensor(
            np.random.RandomState(0).randn(4, 64).astype("float32")
        )
        loss_eager = _train(Net(), x, 50)

        paddle.set_device(dev)
        paddle.seed(2026)
        x = paddle.to_tensor(
            np.random.RandomState(0).randn(4, 64).astype("float32")
        )
        net = paddle.jit.to_static(
            Net(),
            full_graph=True,
            input_spec=[paddle.static.InputSpec([4, 64], "float32")],
        )
        loss_static = _train(net, x, 50)

        np.testing.assert_allclose(loss_static, loss_eager, rtol=1e-5, atol=1e-5)

    def test_cpu_readback_no_race(self):
        # A compiled function returns a device result via .cpu(); the host reads
        # it immediately with no explicit sync. The generic .cpu() copy is
        # blocking, so the snapshot must match a synchronized reference.
        if not self.devs:
            self.skipTest("no custom device available")
        paddle.set_device(f"{self.devs[0]}:0")
        n = 1 << 20
        net = paddle.jit.to_static(
            CpuNet(),
            full_graph=True,
            input_spec=[paddle.static.InputSpec([n], "float32")],
        )
        base = np.random.RandomState(0).randn(n).astype("float32")
        for i in range(200):
            out = net(paddle.to_tensor(base + i)).numpy()
            expect = np.maximum((base + i) * 2.0 - 1.0, 0.0) + 3.0
            np.testing.assert_allclose(out, expect, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
