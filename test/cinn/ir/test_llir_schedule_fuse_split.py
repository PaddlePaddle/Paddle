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

import cinn.schedule as sch
from cinn import to_cinn_llir
from cinn.runtime.data_array import DataArray

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
    @to_cinn_llir
    def elementwise_fuse_assign_loop(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.fuse([i, j, k])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    @to_cinn_llir
    def elementwise_fuse_assign_loop_gt(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(2097152):
            i1 = (i / 128) / 128
            j1 = (i / 128) % 128
            k1 = i % 128
            Y[i1, j1, k1] = 2.0 * X[i1, j1, k1]

    assert_llir_equal(
        elementwise_fuse_assign_loop, elementwise_fuse_assign_loop_gt
    )


def test_split():
    @to_cinn_llir
    def elementwise_split(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.split(i, factors=[2, 1, 64])
                    sch.split(j, factors=[4, 32])
                    sch.split(k, factors=[16, 8])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    @to_cinn_llir
    def elementwise_split_inferred_factor(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.split(i, factors=[-1, 1, 64])
                    sch.split(j, factors=[4, -1])
                    sch.split(k, factors=[-1, 8])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    assert_llir_equal(elementwise_split, elementwise_split_inferred_factor)


def test_split_predicate():
    @to_cinn_llir
    def elementwise_split(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.split(i, factors=[1000, 1, 64])
                    sch.split(j, factors=[4, 32])
                    sch.split(k, factors=[16, 8])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    # comparer = IrCompare(True)
    # comparer.compare(elementwise_split.convert_to_llir().body(),
    #                  elementwise_split_inferred_factor.convert_to_llir().body())
    print(elementwise_split)


def test_fuse_fail_depend_loop():
    @to_cinn_llir
    def elementwise_fuse(
        X: DataArray((128, 128, 128)),
        Y: DataArray((128, 128, 128)),
        Z1: DataArray((128, 128, 128)),
        Z2: DataArray((128, 128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    vi = i
                    vj = j
                    vk = k
                    Z1[vi, vj, vk] = X[vi, vj, vk] * 2.0
                for k in range(128):
                    vi = i
                    vj = j
                    vk = k
                    sch.fuse([j, k])
                    Z2[vi, vj, vk] = X[vi, vj, vk] * 2.0

    print(elementwise_fuse)


def test_fuse_opaque_access():
    @to_cinn_llir
    def opaque_access(X: DataArray((256, 256)), Y: DataArray((256, 256))):
        for i in range(256):
            for j in range(256):
                sch.fuse([i, j])
                vi = i
                vj = j
                X[vi, vj] = 1.0
        for i in range(256):
            for j in range(256):
                sch.fuse([i, j])
                vi = i
                vj = j
                Y[vi, vj] = 2.0

    print(opaque_access)


def test_split_opaque_access():
    @to_cinn_llir
    def opaque_access(X: DataArray((256, 256)), Y: DataArray((256, 256))):
        for i in range(256):
            for j in range(256):
                sch.split(j, factors=[4, -1])
                vi = i
                vj = j
                X[vi, vj] = 1.0
        for i in range(256):
            for j in range(256):
                sch.split(j, factors=[4, -1])
                vi = i
                vj = j
                Y[vi, vj] = 2.0

    print(opaque_access)


def test_fuse_not_affine():
    @to_cinn_llir
    def elementwise_not_affine(
        X: DataArray((128, 128)), Y: DataArray((128, 128))
    ):
        for i in range(4):
            for j in range(126 - i * 32 + 1, 128):
                for k in range(126 - i * 32 + 1, 128):
                    vi = i * 32 + j
                    vj = k
                    sch.fuse([j, k])
                    Y[vi, vj] = X[vi, vj]

    print(elementwise_not_affine)


if __name__ == "__main__":
    test_fuse()
# test_split()
# test_split_predicate()
# test_fuse_fail_depend_loop()
# test_fuse_opaque_access()
# test_split_opaque_access()
# test_fuse_not_affine()
