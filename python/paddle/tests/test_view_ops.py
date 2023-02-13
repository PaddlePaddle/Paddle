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

import unittest

import paddle


class TestViewOpsInplaceVersion(unittest.TestCase):
    def setUp(self):
        pass

    def test_reshape(self):
        a = paddle.randn(shape=[2, 8])
        assert a.inplace_version == 0
        b = a.reshape(shape=[4, 4])
        assert (
            a.inplace_version == 2
        ), f"{a.inplace_version} {b.inplace_version}"
        assert b.inplace_version == 2

    def test_unsqueeze(self):
        a = paddle.randn(shape=[1, 2, 4])
        assert a.inplace_version == 0
        b = a.unsqueeze(0)
        assert (
            a.inplace_version == 2
        ), f"{a.inplace_version} {b.inplace_version}"
        assert b.inplace_version == 2

    def test_squeeze(self):
        a = paddle.randn(shape=[2, 4])
        assert a.inplace_version == 0
        b = a.squeeze(0)
        assert (
            a.inplace_version == 2
        ), f"{a.inplace_version} {b.inplace_version}"
        assert b.inplace_version == 2

    def test_transpose(self):
        a = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b = a.transpose([1, 0])
        assert a.inplace_version == 1
        assert b.inplace_version == 1

    def test_flatten(self):
        a = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b = a.flatten()
        assert (
            a.inplace_version == 2
        ), f"{a.inplace_version} {b.inplace_version}"
        assert (
            b.inplace_version == 2
        ), f"{a.inplace_version} {b.inplace_version}"

    def test_diagonal(self):
        a = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b = a.diagonal()
        assert a.inplace_version == 1
        assert b.inplace_version == 1

    def test_unbind(self):
        a = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b0, b1 = a.unbind()
        assert a.inplace_version == 1
        assert b0.inplace_version == 1
        assert b1.inplace_version == 1

    def test_split(self):
        a = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b0, b1 = a.split(2)
        assert a.inplace_version == 1
        assert b0.inplace_version == 1
        assert b1.inplace_version == 1

    def test_expand(self):
        a = paddle.randn(shape=[1, 2])
        assert a.inplace_version == 0
        b = a.expand([2, 2])
        assert a.inplace_version == 1
        assert b.inplace_version == 1

    def test_expand_as(self):
        a = paddle.randn(shape=[1, 2])
        bb = paddle.randn(shape=[2, 2])
        assert a.inplace_version == 0
        b = a.expand_as(bb)
        assert a.inplace_version == 1
        assert b.inplace_version == 1

    def test_as_real(self):
        a = paddle.randn(shape=[1, 2])
        assert a.inplace_version == 0
        # b = a.as_real()
        # assert a.inplace_version == 1
        # assert b.inplace_version == 1


if __name__ == '__main__':
    unittest.main()
