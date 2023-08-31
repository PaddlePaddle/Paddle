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

# @ to_cinn_llir
# def elementwise_fuse_default_loop(X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))):
#     with sch.fuse():
#         for i in range(128):
#             for j in range(128):
#                 for k in range(128):
#                     Y[i, j, k] = X[i, j, k] * 2.
#


@to_cinn_llir
def elementwise_fuse_assign_loop(
    X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
):
    for i in range(128):
        for j in range(128):
            for k in range(128):
                sch.fuse(i, j, k)
                i1 = i
                j1 = j
                k1 = k
                Y[i1, j1, k1] = X[i1, j1, k1] * 2.0


# @ to_cinn_llir
# def error_case(X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))):
#     for i in range(128):
#         for j in range(128):
#             for k in range(128):
#                 sch.fuse(i, j, k)
#                 Y[i, j, k] = X[i, j, k] * 2.


# @ to_cinn_llir
# def error_case(X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))):
#     for i in range(128):
#         for j in range(128):
#             for k in range(128):
#                 i1, j1, k1 = i, j, k
#                 # sch.fuse(i1, j1, k1)
#                 Y[i1, j1, k1] = X[i1, j1, k1] * 2.

# @ to_cinn_llir
# def error_case(X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))):
#     for i in range(128):
#         for j in range(128):
#             for k in range(128):
#                 i1 = i
#                 j1 = j
#                 k1 = k
#                 sch.fuse(i1, j1, k1)
#                 Y[i1, j1, k1] = X[i1, j1, k1] * 2.


def test_fuse():
    print(elementwise_fuse_assign_loop)
    # assert_equal(elementwise_fuse_default_loop, elementwise_fuse_assign_loop)


if __name__ == "__main__":
    # os.system("read REPLY")
    test_fuse()
