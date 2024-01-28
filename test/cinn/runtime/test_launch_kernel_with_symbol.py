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
def elementwise_add(X, Y, Z, M, N):
    for i in range(M):
        for j in range(N):
            with ir.ScheduleBlockContext("Z"):
                sch.bind(i, "blockIdx.x")
                sch.bind(j, "threadIdx.x")
                vi, vj = ir.AxisMap("SS", [i, j])
                Z[vi, vj] = X[vi, vj] + Y[vi, vj]


def test_launch_kernel_with_symbol():
    M = 2
    N = 4
    x_np = np.random.rand(M, N).astype("float32")
    y_np = np.random.rand(M, N).astype("float32")
    target = cinn.common.DefaultNVGPUTarget()
    x = DataArray.from_numpy(x_np, target)
    y = DataArray.from_numpy(y_np, target)
    z = DataArray.from_numpy(np.zeros_like(x_np), target)
    elementwise_add[target](x, y, z, M, N)
    np.testing.assert_allclose(
        z.to_numpy(), np.add(x_np, y_np), rtol=1e-5, atol=1e-6
    )


# The current Python DSL is not connected to the new executor, and this test is only for local verification of the generated source code
# if __name__ == "__main__":
# test_launch_kernel_with_symbol()
