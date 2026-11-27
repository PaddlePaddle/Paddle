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
        # Tier 2 + Tier 3: scalar CPU pd_op.full (concat axis) and
        # pd_op.full_int_array (reshape) both classify as kInlineMetadata.
        h = paddle.concat([x, y], axis=1)
        h = paddle.tanh(self.fc(h))
        return paddle.reshape(h, [-1, 8])


class ShapeArithmeticNet(nn.Layer):
    def forward(self, x):
        # Tier 2 (shape) + Tier 3 (small static CPU scale on 2-elem shape
        # vector): total I/O numel = 4 <= 64 -> kInlineMetadata -> trace mode.
        s = paddle.shape(x) + 0
        return paddle.reshape(x, s)


class CpuFull64Net(nn.Layer):
    def forward(self, x):
        # Exact Tier 3 threshold (numel == 64) on CPU: still kInlineMetadata.
        c = paddle._C_ops.full([64], 1.0, paddle.float32, paddle.CPUPlace())
        return x + c.to(x.place)


class CpuFull65Net(nn.Layer):
    def forward(self, x):
        # Just above Tier 3 threshold (numel == 65 > 64) on CPU: kCompute ->
        # keeps multi-thread mode.
        c = paddle._C_ops.full([65], 1.0, paddle.float32, paddle.CPUPlace())
        return x + c.to(x.place)


class Tier1BlockingSortNet(nn.Layer):
    def forward(self, x):
        # Tier 1 blocklist: CPU argsort is superlinear/heavy even on a tiny
        # 4-element tensor -> kCompute -> keeps multi-thread mode.
        c = paddle._C_ops.full([4], 1.0, paddle.float32, paddle.CPUPlace())
        idx = paddle.argsort(c)
        return x + idx.astype("float32").to(x.place)


class GraphBudgetExceededNet(nn.Layer):
    def forward(self, x):
        # 17 small CPU full([64]) ops -> cumulative host numel = 1088 > 1024
        # (kGraphInlineMetadataNumelBudget) -> falls back to multi-thread mode.
        acc = x
        for i in range(17):
            c = paddle._C_ops.full(
                [64], float(i + 1) * 0.01, paddle.float32, paddle.CPUPlace()
            )
            acc = acc + c.to(x.place)
        return acc


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


def _run_single_input_case(cls, shape, spec_shape, tag):
    dev = _select_accelerator()
    paddle.set_device(dev)
    paddle.seed(2026)

    net = cls().eval()
    x = paddle.randn(shape, dtype="float32")
    static_net = paddle.jit.to_static(
        net,
        input_spec=[paddle.static.InputSpec(shape=spec_shape, dtype="float32")],
        full_graph=True,
        backend=None,
    )

    with paddle.no_grad():
        ref = net(x).numpy()
        out = static_net(x).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    print(tag, flush=True)


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

    def test_small_static_shape_arithmetic_uses_trace_run(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("shape_arith")
        self.assertIn("SHAPE_ARITH_OK", output)
        self.assertIn("pir interpreter is running by trace mode", output)

    def test_cpu_full_at_threshold_uses_trace_run(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("cpu_full_64")
        self.assertIn("CPU_FULL_64_OK", output)
        self.assertIn("pir interpreter is running by trace mode", output)

    def test_cpu_full_above_threshold_keeps_multithread_mode(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("cpu_full_65")
        self.assertIn("CPU_FULL_65_OK", output)
        self.assertIn("pir interpreter is running by multi-thread mode", output)

    def test_tier1_blocklisted_cpu_op_keeps_multithread_mode(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("tier1_sort")
        self.assertIn("TIER1_SORT_OK", output)
        self.assertIn("pir interpreter is running by multi-thread mode", output)

    def test_graph_budget_exceeded_keeps_multithread_mode(self):
        if _select_accelerator() is None:
            self.skipTest("Requires GPU or CustomDevice accelerator")

        output = _run_self_case("graph_budget")
        self.assertIn("GRAPH_BUDGET_OK", output)
        self.assertIn("pir interpreter is running by multi-thread mode", output)


if __name__ == "__main__":
    case = os.environ.get("TEST_PIR_TRACE_RUN_CASE")
    if case == "lightweight":
        _run_lightweight_case()
    elif case == "shape_arith":
        _run_single_input_case(
            ShapeArithmeticNet, [16, 16], [16, 16], "SHAPE_ARITH_OK"
        )
    elif case == "cpu_full_64":
        _run_single_input_case(CpuFull64Net, [64], [64], "CPU_FULL_64_OK")
    elif case == "cpu_full_65":
        _run_single_input_case(CpuFull65Net, [65], [65], "CPU_FULL_65_OK")
    elif case == "tier1_sort":
        _run_single_input_case(Tier1BlockingSortNet, [4], [4], "TIER1_SORT_OK")
    elif case == "graph_budget":
        _run_single_input_case(
            GraphBudgetExceededNet, [64], [64], "GRAPH_BUDGET_OK"
        )
    else:
        unittest.main()
