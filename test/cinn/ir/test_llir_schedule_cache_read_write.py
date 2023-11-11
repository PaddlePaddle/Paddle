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

from test.cinn.utils.testing import assert_llir_equal

from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch


def test_cache_read_elementwise():
    @to_cinn_llir
    def elementwise_add_cache_read(
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
                with ir.ScheduleBlockContext("B") as B_block:
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        cached_a = sch.cache_read(A_block.block, 0, "global")
        cached_b = sch.cache_read(B_block.block, 0, "local")

    assert_llir_equal(elementwise_add_cache_read, elementwise_add_cache_read)


def test_cache_write_elementwise():
    @to_cinn_llir
    def elementwise_add_cache_write(
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
                with ir.ScheduleBlockContext("B") as B_block:
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    Y[i1, j1] = -A[i1, j1] + 3.0

        cached_a = sch.cache_write(A_block.block, 0, "global")
        cached_b = sch.cache_write(B_block.block, 0, "local")

    # TODO(6clc): core dump
    # assert_llir_equal(elementwise_add_cache_write, elementwise_add_cache_write)


if __name__ == "__main__":
    test_cache_read_elementwise()
    test_cache_write_elementwise()
