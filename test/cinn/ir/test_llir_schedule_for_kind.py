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


# Current Python DSL cannot express the parallel `for`,
# only checks that it can be converted correctly
def test_elementwise_parallel():
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
                    Y[i1, j1] = A[i1, j1] + 2.0
        sch.parallel(A_block.i)

    assert_llir_equal(elementwise_add, elementwise_add)


# Current Python DSL cannot express the vectorize `for`,
# only checks that it can be converted correctly
def test_elementwise_vectorize():
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
            for j0 in range(32):
                for j1 in range(4):
                    with ir.ScheduleBlockContext("Y") as Y_block:
                        i1, j1 = ir.AxisMap("SS", [i, j0 * 4 + j1])
                        Y[i1, j1] = A[i1, j1] + 2.0
        sch.vectorize(Y_block.j1, 1)

    assert_llir_equal(elementwise_add, elementwise_add)


# Current Python DSL cannot express the unroll `for`,
# only checks that it can be converted correctly
def test_elementwise_unroll():
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
            for j0 in range(32):
                for j1 in range(4):
                    with ir.ScheduleBlockContext("Y") as Y_block:
                        i1, j1 = ir.AxisMap("SS", [i, j0 * 4 + j1])
                        Y[i1, j1] = A[i1, j1] + 2.0
        sch.unroll(Y_block.j1)

    assert_llir_equal(elementwise_add, elementwise_add)


if __name__ == "__main__":
    test_elementwise_parallel()
    test_elementwise_vectorize()
    test_elementwise_unroll()
