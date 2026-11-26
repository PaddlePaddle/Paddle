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

import os
import subprocess
import sys
import unittest

import numpy as np

import paddle
from paddle import nn


def _select_accelerator():
    if (
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
        return "gpu:0"
    custom_types = paddle.device.get_all_custom_device_type() or []
    if custom_types:
        return f"{custom_types[0]}:0"
    return None


class ConcatReshapeNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 8)

    def forward(self, x, y):
        # Emits lightweight host metadata ops in PIR (scalar CPU pd_op.full for
        # concat axis, pd_op.full_int_array, and pd_op.shape).
        h = paddle.concat([x, y], axis=1)
        h = paddle.tanh(self.fc(h))
        return paddle.reshape(h, [-1, 8])


class NonLightweightCpuOpNet(nn.Layer):
    def forward(self, x):
        # Computing on a host shape tensor emits a non-lightweight CPU op
        # (pd_op.add on CPU), which must keep multi-thread mode active.
        s = paddle.shape(x) + 0
        return paddle.reshape(x, s)


def _run_lightweight_case():
    dev = _select_accelerator()
    paddle.set_device(dev)
    paddle.seed(2026)

    net = ConcatReshapeNet().eval()
    x = paddle.randn([64, 1], dtype="float32")
    y = paddle.randn([64, 1], dtype="float32")

    backend = "CINN" if paddle.is_compiled_with_cinn() else None
    static_net = paddle.jit.to_static(
        net,
        input_spec=[
            paddle.static.InputSpec(shape=[64, 1], dtype="float32"),
            paddle.static.InputSpec(shape=[64, 1], dtype="float32"),
        ],
        full_graph=True,
        backend=backend,
    )

    with paddle.no_grad():
        ref = net(x, y).numpy()
        out = static_net(x, y).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    print("LIGHTWEIGHT_CASE_OK", flush=True)


def _run_non_lightweight_case():
    dev = _select_accelerator()
    paddle.set_device(dev)
    paddle.seed(2026)

    net = NonLightweightCpuOpNet().eval()
    x = paddle.randn([16, 16], dtype="float32")
    static_net = paddle.jit.to_static(
        net,
        input_spec=[paddle.static.InputSpec(shape=[16, 16], dtype="float32")],
        full_graph=True,
        backend=None,
    )

    with paddle.no_grad():
        ref = net(x).numpy()
        out = static_net(x).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    print("NON_LIGHTWEIGHT_CASE_OK", flush=True)


def _run_self_case(case_name):
    env = os.environ.copy()
    env.pop("FLAGS_enable_pir_in_executor_trace_run", None)
    env["FLAGS_enable_pir_api"] = "1"
    env["TEST_PIR_TRACE_RUN_CASE"] = case_name
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout + "\n" + proc.stderr


class TestPirTraceRunLightweightSyncOps(unittest.TestCase):
    def test_lightweight_metadata_sync_ops_use_trace_run(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("lightweight")
        self.assertIn("LIGHTWEIGHT_CASE_OK", output)
        self.assertIn("pir interpreter is running by trace mode", output)

    def test_non_lightweight_cpu_op_keeps_multithread_mode(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("non_lightweight")
        self.assertIn("NON_LIGHTWEIGHT_CASE_OK", output)
        self.assertIn("pir interpreter is running by multi-thread mode", output)


if __name__ == "__main__":
    case = os.environ.get("TEST_PIR_TRACE_RUN_CASE")
    if case == "lightweight":
        _run_lightweight_case()
    elif case == "non_lightweight":
        _run_non_lightweight_case()
    else:
        unittest.main()
