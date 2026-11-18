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

import time
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


def _run(dev, force_sync, steps, warmup):
    paddle.set_device(dev)
    paddle.seed(2026)
    paddle.set_flags({"FLAGS_custom_device_force_run_sync": force_sync})
    net = paddle.jit.to_static(
        Net(),
        full_graph=True,
        input_spec=[paddle.static.InputSpec([4, 64], "float32")],
    )
    opt = paddle.optimizer.SGD(
        learning_rate=0.01, parameters=net.parameters()
    )
    x = paddle.to_tensor(
        np.random.RandomState(0).randn(4, 64).astype("float32")
    )
    losses, times = [], []
    for i in range(steps):
        paddle.device.synchronize()
        t = time.perf_counter()
        loss = net(x).mean()
        loss.backward()
        opt.step()
        opt.clear_grad()
        paddle.device.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t)
        losses.append(float(loss))
    return np.array(losses), float(np.median(times) * 1000)


class TestCustomDeviceRunBoundary(unittest.TestCase):
    def setUp(self):
        self.devs = paddle.device.get_all_custom_device_type() or []

    def test_correctness_and_perf(self):
        if not self.devs:
            self.skipTest("no custom device available")
        dev = f"{self.devs[0]}:0"
        steps, warmup = 50, 10
        loss_sync, ms_sync = _run(dev, True, steps, warmup)
        loss_async, ms_async = _run(dev, False, steps, warmup)
        # async event ordering must be numerically identical to blocking sync
        np.testing.assert_allclose(loss_async, loss_sync, rtol=0, atol=0)
        # async path must not be slower than the blocking full-device sync
        self.assertLessEqual(ms_async, ms_sync * 1.15)
        print(f"[run-boundary] sync={ms_sync:.3f}ms async={ms_async:.3f}ms")


if __name__ == "__main__":
    unittest.main()
