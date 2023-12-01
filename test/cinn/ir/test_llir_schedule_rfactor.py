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


def test_matmul():
    @to_cinn_llir
    def matmul(
        A: DataArray((128, 128)),
        B: DataArray((128, 128)),
        C: DataArray((128, 128)),
    ):
        for i0 in range(128):
            for i1 in range(128):
                with ir.ScheduleBlockContext("init"):
                    vi, vj = ir.AxisMap("SS", [i0, i1])
                    C[vi, vj] = 0.0
                for i2_outer in range(4):
                    for i2_inner_outer in range(8):
                        for i2_inner_inner in range(4):
                            with ir.ScheduleBlockContext(
                                "compute"
                            ) as Compute_block:
                                vi, vj, vk = ir.AxisMap(
                                    "SSR",
                                    [
                                        i0,
                                        i1,
                                        i2_outer * 32
                                        + i2_inner_outer * 4
                                        + i2_inner_inner,
                                    ],
                                )
                                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])
        sch.rfactor(Compute_block.i2_inner_inner, 0)

    # TODO(6clc): rfactor schedule rasie Error Message: iter_value not support complex reduce bindings
    # assert_llir_equal(matmul, matmul)


if __name__ == "__main__":
    test_matmul()
