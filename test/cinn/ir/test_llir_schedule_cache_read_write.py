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


from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch


# (Note:LiuYang): Here the temp tensor is created in cache_read or cache_write
# so that the two ir is not equal and we just judge them by string of them
def test_cache_read_elementwise():
    class origin:
        @to_cinn_llir
        def elementwise_add_cache_read(
            X: DataArray((128, 128)),
            Y: DataArray((128, 128)),
            A: DataArray((128, 128)),
            A_local_temp_buffer: DataArray((128, 128)),
            N: ir.Var(),
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

            cached_b = sch.cache_read(B_block.block, 0, "local")

    class expected:
        @to_cinn_llir
        def elementwise_add_cache_read(
            X: DataArray((128, 128)),
            Y: DataArray((128, 128)),
            A: DataArray((128, 128)),
            A_local_temp_buffer: DataArray((128, 128)),
            N: ir.Var(),
        ):
            for i in range(128):
                for j in range(128):
                    with ir.ScheduleBlockContext("A") as A_block:
                        i1, j1 = ir.AxisMap("SS", [i, j])
                        A[i1, j1] = X[i1, j1] * 2.0
            for cache_ax0 in range(128):
                for cache_ax1 in range(128):
                    with ir.ScheduleBlockContext(
                        "A_local_temp_buffer"
                    ) as A_local_temp_buffer_block:
                        v0, v1 = ir.AxisMap("SS", [cache_ax0, cache_ax1])
                        A_local_temp_buffer[v0, v1] = A[v0, v1]
            for i3 in range(128):
                for j3 in range(128):
                    with ir.ScheduleBlockContext("B") as B_block:
                        i1, j1 = ir.AxisMap("SS", [i3, j3])
                        Y[i1, j1] = -A_local_temp_buffer[i1, j1] + 3.0


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


def test_cache_read_elementwise_dynamic():
    class origin:
        @to_cinn_llir
        def elementwise_add_cache_read(
            X: DataArray((-1, 128)),
            Y: DataArray((-1, 128)),
            A: DataArray((-1, 128)),
            A_local_temp_buffer: DataArray((-1, 128)),
            N: ir.Var(),
        ):
            for i in range(N):
                for j in range(128):
                    with ir.ScheduleBlockContext("A") as A_block:
                        i1, j1 = ir.AxisMap("SS", [i, j])
                        A[i1, j1] = X[i1, j1] * 2.0
            for i3 in range(N):
                for j3 in range(128):
                    with ir.ScheduleBlockContext("B") as B_block:
                        i1, j1 = ir.AxisMap("SS", [i3, j3])
                        Y[i1, j1] = -A[i1, j1] + 3.0

            cached_b = sch.cache_read(B_block.block, 0, "local")

    class expected:
        @to_cinn_llir
        def elementwise_add_cache_read(
            X: DataArray((-1, 128)),
            Y: DataArray((-1, 128)),
            A: DataArray((-1, 128)),
            A_local_temp_buffer: DataArray((-1, 128)),
            N: ir.Var(),
        ):
            for i in range(N):
                for j in range(128):
                    with ir.ScheduleBlockContext("A") as A_block:
                        i1, j1 = ir.AxisMap("SS", [i, j])
                        A[i1, j1] = X[i1, j1] * 2.0
            for cache_ax0 in range(N):
                for cache_ax1 in range(128):
                    with ir.ScheduleBlockContext(
                        "A_local_temp_buffer"
                    ) as A_local_temp_buffer_block:
                        v0, v1 = ir.AxisMap("SS", [cache_ax0, cache_ax1])
                        A_local_temp_buffer[v0, v1] = A[v0, v1]
            for i3 in range(N):
                for j3 in range(128):
                    with ir.ScheduleBlockContext("B") as B_block:
                        i1, j1 = ir.AxisMap("SS", [i3, j3])
                        Y[i1, j1] = -A_local_temp_buffer[i1, j1] + 3.0

    assert str(origin.elementwise_add_cache_read) == str(
        expected.elementwise_add_cache_read
    )


if __name__ == "__main__":
    test_cache_read_elementwise()
    test_cache_write_elementwise()
