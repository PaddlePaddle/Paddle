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


import cinn
import numpy as np
from cinn import ir, to_cinn_llir
from cinn.runtime.data_array import DataArray
from cinn.schedule import IRSchedule as sch


@to_cinn_llir
def reduce_max(A, B):
    for i1 in range(1):
        for j1 in range(2):
            for k1 in range(4):
                with ir.ScheduleBlockContext("init") as init:
                    vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                    B[vi, vj, vk] = 0.0
                for l1 in range(8):
                    with ir.ScheduleBlockContext("B"):
                        sch.bind(i1, "blockIdx.x")
                        sch.bind(j1, "threadIdx.y")
                        sch.bind(k1, "threadIdx.x")
                        vi1, vj1, vk1, vl1 = ir.AxisMap(
                            "SSSR", [i1, j1, k1, l1]
                        )
                        B[vi1, vj1, vk1] = ir.Max.make(
                            B[vi1, vj1, vk1], A[vi1, vj1, vk1, vl1]
                        )


@to_cinn_llir
def reduce_sum(A, B):
    for i1 in range(1):
        for j1 in range(2):
            for k1 in range(4):
                with ir.ScheduleBlockContext("init") as init:
                    vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                    B[vi, vj, vk] = 0.0
                for l1 in range(8):
                    with ir.ScheduleBlockContext("B"):
                        sch.bind(i1, "blockIdx.x")
                        sch.bind(j1, "threadIdx.y")
                        sch.bind(k1, "threadIdx.x")
                        vi1, vj1, vk1, vl1 = ir.AxisMap(
                            "SSSR", [i1, j1, k1, l1]
                        )
                        B[vi1, vj1, vk1] = (
                            B[vi1, vj1, vk1] + A[vi1, vj1, vk1, vl1]
                        )


def test_reduce_max_cuda():
    # prepare input and output array
    d1 = 2
    d2 = 4
    d3 = 8
    a_np = np.random.rand(1, d1, d2, d3).astype("float32")
    b_np = a_np.max(axis=-1).astype("float32")
    target = cinn.common.DefaultNVGPUTarget()
    a = DataArray.from_numpy(a_np, target)
    b = DataArray.from_numpy(np.zeros_like(b_np), target)
    reduce_max[target](a, b)
    np.testing.assert_allclose(b.to_numpy(), b_np, rtol=1e-5, atol=1e-6)


def test_reduce_sum_cuda():
    # prepare input and output array
    d1 = 2
    d2 = 4
    d3 = 8
    a_np = np.random.rand(1, d1, d2, d3).astype("float32")
    b_np = a_np.sum(axis=-1).astype("float32")
    target = cinn.common.DefaultNVGPUTarget()
    a = DataArray.from_numpy(a_np, target)
    b = DataArray.from_numpy(np.zeros_like(b_np), target)
    reduce_sum[target](a, b)
    np.testing.assert_allclose(b.to_numpy(), b_np, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_reduce_max_cuda()
    test_reduce_sum_cuda()
