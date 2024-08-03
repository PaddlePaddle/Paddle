# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.cinn import ir, to_cinn_llir
from paddle.cinn.runtime.data_array import DataArray
from paddle.cinn.schedule import IRSchedule as sch
from test.cinn.utils.testing import assert_llir_equal


def test_compute_at_elementwise():
    @to_cinn_llir
    def elementwise_add(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A") as A_block:
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    sch.compute_at(A_block.block, i, False)
                    Y[i1, j1] = A[i1, j1] + 2.0

    @to_cinn_llir
    def elementwise_add_gt(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, 0 + j])
                    A[i1, j1] = X[i1, j1] * 2.0
            for k in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i2, k1 = ir.AxisMap("SS", [i, k])
                    Y[i2, k1] = A[i2, k1] + 2.0

    assert_llir_equal(elementwise_add, elementwise_add_gt)


def test_reverse_compute_at():
    @to_cinn_llir
    def reverse_compute_at_tiled(
        A: DataArray((128, 128)),
        B: DataArray((128, 128)),
        C: DataArray((128, 128)),
    ):
        for i0 in range(8):
            for j0 in range(8):
                for i1 in range(16):
                    for j1 in range(16):
                        with ir.ScheduleBlockContext("B") as B_block:
                            vi, vj = ir.AxisMap(
                                "SS", [i0 * 16 + i1, j0 * 16 + j1]
                            )
                            B[vi, vj] = A[vi, vj] * 2.0
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("C") as C_block:
                    vi, vj = ir.AxisMap("SS", [i, j])
                    C[vi, vj] = B[vi, vj] + 1.0

        sch.reverse_compute_at(C_block.block, B_block.i1)

    @to_cinn_llir
    def reverse_compute_at_tiled_gt(
        A: DataArray((128, 128)),
        B: DataArray((128, 128)),
        C: DataArray((128, 128)),
    ):
        for i0 in range(8):
            for j0 in range(8):
                for i1 in range(16):
                    for j1 in range(16):
                        with ir.ScheduleBlockContext("B") as B_block:
                            vi, vj = ir.AxisMap(
                                "SS", [i0 * 16 + i1, j0 * 16 + j1]
                            )
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j2 in range(16):
                        with ir.ScheduleBlockContext("C") as C_block:
                            vi, vj = ir.AxisMap(
                                "SS", [16 * i0 + i1, 16 * j0 + j2]
                            )
                            C[vi, vj] = B[vi, vj] + 1.0

    assert_llir_equal(reverse_compute_at_tiled, reverse_compute_at_tiled_gt)


def test_compute_at_elementwise_dynamic():
    @to_cinn_llir
    def elementwise_add(
        X: DataArray((-1, 128)),
        Y: DataArray((-1, 128)),
        A: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("A") as A_block:
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    sch.compute_at(A_block.block, i, False)
                    Y[i1, j1] = A[i1, j1] + 2.0

    @to_cinn_llir
    def elementwise_add_gt(
        X: DataArray((-1, 128)),
        Y: DataArray((-1, 128)),
        A: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, 0 + j])
                    A[i1, j1] = X[i1, j1] * 2.0
            for k in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i2, k1 = ir.AxisMap("SS", [i, k])
                    Y[i2, k1] = A[i2, k1] + 2.0

    assert_llir_equal(elementwise_add, elementwise_add_gt)


def test_reverse_compute_at_dynamic():
    @to_cinn_llir
    def reverse_compute_at_tiled(
        A: DataArray((-1, 128)),
        B: DataArray((-1, 128)),
        C: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i0 in range(N / 16):
            for j0 in range(8):
                for i1 in range(16):
                    for j1 in range(16):
                        with ir.ScheduleBlockContext("B") as B_block:
                            vi, vj = ir.AxisMap(
                                "SS", [i0 * 16 + i1, j0 * 16 + j1]
                            )
                            B[vi, vj] = A[vi, vj] * 2.0
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("C") as C_block:
                    vi, vj = ir.AxisMap("SS", [i, j])
                    C[vi, vj] = B[vi, vj] + 1.0

        sch.reverse_compute_at(C_block.block, B_block.i1)

    @to_cinn_llir
    def reverse_compute_at_tiled_gt(
        A: DataArray((-1, 128)),
        B: DataArray((-1, 128)),
        C: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i0 in range(N / 16):
            for j0 in range(8):
                for i1 in range(16):
                    for j1 in range(16):
                        with ir.ScheduleBlockContext("B") as B_block:
                            vi, vj = ir.AxisMap(
                                "SS", [i0 * 16 + i1, j0 * 16 + j1]
                            )
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j2 in range(16):
                        with ir.ScheduleBlockContext("C") as C_block:
                            vi, vj = ir.AxisMap(
                                "SS", [16 * i0 + i1, 16 * j0 + j2]
                            )
                            C[vi, vj] = B[vi, vj] + 1.0

    assert_llir_equal(reverse_compute_at_tiled, reverse_compute_at_tiled_gt)


if __name__ == '__main__':
    test_compute_at_elementwise()
    test_reverse_compute_at()
    test_compute_at_elementwise_dynamic()
    test_reverse_compute_at_dynamic()
