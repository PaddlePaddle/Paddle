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

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle


def tensor_method_call_1(x: paddle.Tensor):
    y = x + 1
    return y.mean()


def tensor_method_call_2(a: paddle.Tensor, b: paddle.Tensor):
    c = a.add(b)
    d = c.multiply(a)
    e = d.subtract(b)
    f = e.divide(a)
    g = f.pow(2) + f.abs().sqrt()
    h = (g.abs() + 1).log() - (g / g.max()).exp()
    i = h.sin() + h.cos()
    return i


def tensor_method_passed_by_user(a: paddle.Tensor, func: paddle.Tensor):
    return func(a)


def tensor_method_property(a: paddle.Tensor, b: paddle.Tensor):
    return (
        a.name,
        str(a.place),
        a.persistable,
        a.dtype,
        a.type,
        a.is_tensor(),
        a.clear_gradient(),
        a @ b.T.astype(a.dtype)
        + len(a.shape)
        + b.size
        + a.ndim
        + a.dim()
        + a.rank(),
    )


def tensor_method_property_mT(a: paddle.Tensor):
    return a.mT


def middle_tensor_name(a: paddle.Tensor, b: paddle.Tensor):
    c = a + b
    return c.name


class TestTensorMethod(TestCaseBase):
    def test_tensor_method_1(self):
        x = paddle.rand([10])
        y = paddle.rand([2, 4, 6])
        self.assert_results(tensor_method_call_1, x)
        self.assert_results(tensor_method_call_1, y)

    def test_tensor_method_2(self):
        x = paddle.rand([42])
        y = paddle.rand([42])
        self.assert_results(tensor_method_call_2, x, y)

    def test_tensor_method_passed_by_user(self):
        x = paddle.rand([42])
        y = paddle.rand([42])
        self.assert_results(tensor_method_passed_by_user, x, y.add)

    @test_with_faster_guard
    def test_tensor_method_property(self):
        x = paddle.rand([42, 24], dtype='float64')
        y = paddle.rand([42, 24], dtype='float32')
        self.assert_results(tensor_method_property, x, y)

    @unittest.skip("TODO: dynamic tensor name is different")
    def test_middle_tensor_name(self):
        x = paddle.rand([42, 24])
        y = paddle.rand([42, 24])
        self.assert_results(middle_tensor_name, x, y)

    def test_tensor_method_property_mT(self):
        x = paddle.rand([42, 24, 2, 2, 3, 2], dtype='float64')
        y = paddle.rand([42, 24, 2, 3, 3, 2], dtype='float32')
        self.assert_results(tensor_method_property_mT, x)
        self.assert_results(tensor_method_property_mT, y)


if __name__ == "__main__":
    unittest.main()
