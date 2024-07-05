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

from paddle.cinn import common, ir, to_cinn_llir
from paddle.cinn.runtime.data_array import DataArray
from paddle.cinn.schedule import IRSchedule as sch
from test.cinn.utils.testing import assert_llir_equal


def test_compute_inline_elementwise():
    @to_cinn_llir
    def elementwise_add_inline(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A") as A_block:
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i3 in range(128):
            for j3 in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        block_a = sch.get_block("A")
        sch.compute_inline(block_a)

    @to_cinn_llir
    def elementwise_add_inline_gt(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    Y[i1, j1] = -(X[i1, j1] * 2.0) + 3.0

    assert_llir_equal(elementwise_add_inline, elementwise_add_inline_gt)


def test_reverse_compute_inline_elementwise():
    @to_cinn_llir
    def elementwise_add_inline(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A") as A_block:
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i3 in range(128):
            for j3 in range(128):
                with ir.ScheduleBlockContext("Y") as Y_block:
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        sch.reverse_compute_inline(Y_block.block)

    @to_cinn_llir
    def elementwise_add_inline_gt(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    Y[i1, j1] = -(X[i1, j1] * 2.0) + 3.0

    assert_llir_equal(elementwise_add_inline, elementwise_add_inline_gt)


def test_compute_inline_elementwise_dynamic():
    @to_cinn_llir
    def elementwise_add_inline(
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
        for i3 in range(N):
            for j3 in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        block_a = sch.get_block("A")
        sch.compute_inline(block_a)

    @to_cinn_llir
    def elementwise_add_inline_gt(
        X: DataArray((-1, 128)),
        Y: DataArray((-1, 128)),
        A: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    Y[i1, j1] = -(X[i1, j1] * 2.0) + 3.0

    assert_llir_equal(elementwise_add_inline, elementwise_add_inline_gt)


def test_reverse_compute_inline_elementwise_dynamic():
    @to_cinn_llir
    def elementwise_add_inline(
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
        for i3 in range(-1):
            for j3 in range(128):
                with ir.ScheduleBlockContext("Y") as Y_block:
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        sch.reverse_compute_inline(Y_block.block)

    @to_cinn_llir
    def elementwise_add_inline_gt(
        X: DataArray((-1, 128)),
        Y: DataArray((-1, 128)),
        A: DataArray((-1, 128)),
        N: ir.Var(),
    ):
        for i in range(N):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    Y[i1, j1] = -(X[i1, j1] * 2.0) + 3.0

    assert_llir_equal(elementwise_add_inline, elementwise_add_inline_gt)


def test_gather_slice_concat_mul_pattern_compute_inline():
    class origin:
        @to_cinn_llir
        def fn_rotary(
            A: DataArray((1, -1, 1, 512)),
            var_1: DataArray((1, -1, 1, 256)),
            var_2: DataArray((1, -1, 1, 256)),
            var_3: DataArray((1, -1, 1, 512)),
            C: DataArray((1, -1, 1, 512)),
            index: DataArray((2, 1), dtype=common.Int(64)),
            Out: DataArray((1, -1, 1, 512)),
            seq_len: ir.Var(),
        ):
            for i in range(0, 1):
                for j in range(0, seq_len):
                    for k in range(0, 1):
                        for a in range(0, 256):
                            with ir.ScheduleBlockContext(
                                "var_1"
                            ) as var_1_block:
                                i0, i1, i2, i3 = ir.AxisMap(
                                    "SSSS", [i, j, k, a]
                                )
                                var_1[i0, i1, i2, i3] = A[
                                    ir.Expr(index[i0, ir.Expr(0)]), i1, i2, i3
                                ]
            for i in range(0, 1):
                for j in range(0, seq_len):
                    for k in range(0, 1):
                        for a in range(0, 256):
                            with ir.ScheduleBlockContext(
                                "var_2"
                            ) as var_2_block:
                                i0, i1, i2, i3 = ir.AxisMap(
                                    "SSSS", [i, j, k, a]
                                )
                                var_2[i0, i1, i2, i3] = A[
                                    ir.Expr(index[i0 + ir.Expr(1), ir.Expr(0)]),
                                    i1,
                                    i2,
                                    ir.Expr(256) + i3,
                                ]
            for i in range(0, 1):
                for j in range(0, seq_len):
                    for k in range(0, 1):
                        for a in range(0, 512):
                            with ir.ScheduleBlockContext(
                                "var_3"
                            ) as var_3_block:
                                i0, i1, i2, i3 = ir.AxisMap(
                                    "SSSS", [i, j, k, a]
                                )
                                var_3[i0, i1, i2, i3] = ir.Select.make(
                                    (i3 < 256),
                                    var_1[i0, i1, i2, i3],
                                    var_2[i0, i1, i2, (i3 - ir.Expr(256))],
                                )
            for i in range(0, 1):
                for j in range(0, seq_len):
                    for k in range(0, 1):
                        for a in range(0, 512):
                            with ir.ScheduleBlockContext("out") as out_block:
                                i0, i1, i2, i3 = ir.AxisMap(
                                    "SSSS", [i, j, k, a]
                                )
                                Out[i0, i1, i2, i3] = (
                                    C[i0, i1, i2, i3] * var_3[i0, i1, i2, i3]
                                )
            block_var_1 = sch.get_block("var_1")
            sch.compute_inline(block_var_1)
            block_var_2 = sch.get_block("var_2")
            sch.compute_inline(block_var_2)
            block_var_3 = sch.get_block("var_3")
            sch.compute_inline(block_var_3)

    class expected:
        @to_cinn_llir
        def fn_rotary(
            A: DataArray((1, -1, 1, 512)),
            C: DataArray((1, -1, 1, 512)),
            index: DataArray((2, 1), dtype=common.Int(64)),
            Out: DataArray((1, -1, 1, 512)),
            seq_len: ir.Var(),
        ):
            for i in range(0, 1):
                for j in range(0, seq_len):
                    for k in range(0, 1):
                        for a in range(0, 512):
                            with ir.ScheduleBlockContext("out") as out_block:
                                i0, i1, i2, i3 = ir.AxisMap(
                                    "SSSS", [i, j, k, a]
                                )
                                Out[i0, i1, i2, i3] = C[
                                    i0, i1, i2, i3
                                ] * ir.Select.make(
                                    (i3 < 256),
                                    A[
                                        ir.Expr(index[i0, ir.Expr(0)]),
                                        i1,
                                        i2,
                                        i3,
                                    ],
                                    A[
                                        ir.Expr(
                                            index[(i0 + ir.Expr(1)), ir.Expr(0)]
                                        ),
                                        i1,
                                        i2,
                                        ir.Expr(256) + (i3 - ir.Expr(256)),
                                    ],
                                )

    assert_llir_equal(origin.fn_rotary, expected.fn_rotary)


if __name__ == "__main__":
    test_compute_inline_elementwise()
    test_reverse_compute_inline_elementwise()
    test_compute_inline_elementwise_dynamic()
    test_reverse_compute_inline_elementwise_dynamic()
    test_gather_slice_concat_mul_pattern_compute_inline()
