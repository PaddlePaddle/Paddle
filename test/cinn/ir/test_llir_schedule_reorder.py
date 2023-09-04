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

import cinn.schedule as sch
from cinn import to_cinn_llir
from cinn.runtime.data_array import DataArray


def test_reorder_elementwise():
    @to_cinn_llir
    def reorder_elementwise(
        X: DataArray((64, 64, 64, 64)), Y: DataArray((64, 64, 64, 64))
    ):
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    for l in range(8):
                        sch.reorder([k, l, i])
                        vi = i
                        vj = j
                        vk = k
                        vl = 8 * l
                        Y[vi, vj, vk, vk] = X[vi, vj, vk, vl] * 2.0

    print(reorder_elementwise)


def test_reorder_overlapped():
    @to_cinn_llir
    def reorder_overlapped(X: DataArray((28, 8)), Y: DataArray((28, 8))):
        for i in range(12):
            for j in range(4):
                for k in range(4):
                    sch.reorder([i, k, j])
                    vi = i * 2 + j
                    vj = k
                    Y[vi, vj] = X[vi, vj] + 1.0

    print(reorder_overlapped)


if __name__ == '__main__':
    test_reorder_elementwise()
    test_reorder_overlapped()
