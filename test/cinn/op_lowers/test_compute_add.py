#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from test.cinn.utils.testing import assert_llir_equal
from cinn import ir, to_cinn_llir
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder
from cinn.hlir.transform import LowerOpPass
from cinn.runtime.data_array import DataArray


def test_add():
    builder = NetBuilder("test_basic")
    a = builder.create_input(Float(32), ["n"], "A", "")
    b = builder.create_input(Float(32), ["n"], "B", "")
    c = builder.add(a, b)
    prog = builder.build()

    @to_cinn_llir
    def bin_op_kernel(
        A: DataArray("n"), B: DataArray("n"), var_1: DataArray("n"), n: int
    ):
        for i in range(n):
            with ir.ScheduleBlockContext("var_1"):
                i0 = ir.AxisMap("S", [i])
                var_1[i0] = A[i0] + B[i0]

    target = DefaultNVGPUTarget()
    bin_compute_op = LowerOpPass(prog, target)

    assert_llir_equal(bin_compute_op[0], bin_compute_op[0])
    bin_compute_op[target]


def test_add_e2e():
    target = cinn.common.DefaultNVGPUTarget()
    builder = NetBuilder("test_basic")
    a = builder.create_input(Float(32), ["n"], "A", "")
    b = builder.create_input(Float(32), ["n"], "B", "")
    c = builder.add(a, b)
    prog = builder.build()

    # Lower
    bin_compute_op = LowerOpPass(prog, target)
    # Compile
    run_module = cinn.compile(bin_compute_op, target, {"A", "B", "var_1"})

    N = 10
    X_np = np.random.random(N).astype(np.float32)
    Y_np = np.random.random(N).astype(np.float32)
    Z_np = np.zeros((N), dtype=np.float32)
    X = DataArray.from_numpy(X_np, target)
    Y = DataArray.from_numpy(Y_np, target)
    Z = DataArray.from_numpy(Z_np, target)

    run_module(X, Y, Z, N)
    pred = Z.to_numpy()
    gt = np.add(X_np, Y_np)
    np.testing.assert_allclose(pred, gt)


test_add()
