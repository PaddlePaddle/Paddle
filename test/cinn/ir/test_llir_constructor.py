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


from cinn import ir, lang, to_cinn_llir
from cinn.runtime.data_array import DataArray


def test_call_extern():
    @to_cinn_llir
    def call_sinh(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockContext("init") as init:
                        vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                        B[vi, vj, vk] = lang.call_extern(
                            "sinh", [A[vi, vi, vj, vk]], {}
                        )

    str(call_sinh)


if __name__ == "__main__":
    test_call_extern()
