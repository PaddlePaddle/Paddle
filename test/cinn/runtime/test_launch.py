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
from cinn import to_cinn_llir
from cinn.runtime.data_array import DataArray


@to_cinn_llir
def add_kernel(X, Y, Z, N):
    for idx in range(N):
        Z[idx] = X[idx] + Y[idx]


def test_launch():
    N = 32
    X_np = np.random.random(N).astype(np.float32)
    Y_np = np.random.random(N).astype(np.float32)
    Z_np = np.zeros((N), dtype=np.float32)
    target = cinn.common.DefaultNVGPUTarget()
    X = DataArray.from_numpy(X_np, target)
    Y = DataArray.from_numpy(Y_np, target)
    Z = DataArray.from_numpy(Z_np, target)

    # compile and run
    add_kernel[target](X, Y, Z, N)
    pred = Z.to_numpy()
    gt = np.add(X_np, Y_np)
    np.testing.assert_allclose(pred, gt)


if __name__ == "__main__":
    test_launch()
