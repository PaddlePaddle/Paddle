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

import operator
import unittest

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle


def unary_positive(x: int):
    y = +x
    return y


def unary_negative(x: paddle.Tensor):
    y = -x
    return y


def unary_not(x: paddle.Tensor):
    y = not x
    return y


def unary_invert(x: paddle.Tensor):
    y = ~x
    return y


def binary_power(x: paddle.Tensor, y: paddle.Tensor):
    z = x**y
    return z


def binary_multiply(x: paddle.Tensor, y: paddle.Tensor):
    z = x * y
    return z


def binary_matrix_multiply(x: paddle.Tensor, y: paddle.Tensor):
    z = x @ y
    return z


def binary_floor_divide(x: paddle.Tensor, y: paddle.Tensor):
    z = x // y
    return z


def binary_true_divide(x: paddle.Tensor, y: paddle.Tensor):
    z = x / y
    return z


def binary_modulo(x: paddle.Tensor, y: paddle.Tensor):
    z = x % y
    return z


def binary_add(x: paddle.Tensor, y: paddle.Tensor):
    z = x + y
    return z


def binary_subtract(x: paddle.Tensor, y: paddle.Tensor):
    z = x - y
    return z


def binary_lshift(x: int, y: int):
    z = x << y
    return z


def binary_rshift(x: int, y: int):
    z = x >> y
    return z


def binary_and(x: paddle.Tensor, y: paddle.Tensor):
    z = x & y
    return z


def binary_or(x: paddle.Tensor, y: paddle.Tensor):
    z = x | y
    return z


def binary_xor(x: paddle.Tensor, y: paddle.Tensor):
    z = x ^ y
    return z


def inplace_power(x: paddle.Tensor, y: paddle.Tensor):
    x **= y
    return x


def inplace_multiply(x: paddle.Tensor, y: paddle.Tensor):
    x *= y
    return x


def inplace_matrix_multiply(x: paddle.Tensor, y: paddle.Tensor):
    x @= y
    return x


def inplace_floor_divide(x: paddle.Tensor, y: paddle.Tensor):
    x //= y
    return x


def inplace_true_divide(x: paddle.Tensor, y: paddle.Tensor):
    x /= y
    return x


def inplace_modulo(x: paddle.Tensor, y: paddle.Tensor):
    x %= y
    return x


def inplace_add(x: paddle.Tensor, y: paddle.Tensor):
    x += y
    return x


def inplace_subtract(x: paddle.Tensor, y: paddle.Tensor):
    x -= y
    return x


def inplace_lshift(x: paddle.Tensor, y: int):
    x <<= y
    return x


def inplace_rshift(x: paddle.Tensor, y: int):
    x >>= y
    return x


def inplace_and(x: paddle.Tensor, y: paddle.Tensor):
    x &= y
    return x


def inplace_or(x: paddle.Tensor, y: paddle.Tensor):
    x |= y
    return x


def inplace_xor(x: paddle.Tensor, y: paddle.Tensor):
    x ^= y
    return x


def list_getitem(x: int, y: paddle.Tensor):
    z = [x, y]
    return operator.getitem(z, 1) + 1


def list_getitem_slice(x: int, y: paddle.Tensor):
    z = [x, y]
    return operator.getitem(z, slice(0, 2))


def list_setitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.setitem(z, 0, 3)
    return z


def list_setitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.setitem(z, 1, paddle.to_tensor(3))
    return z


def list_delitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.delitem(z, 0)
    return z


def list_delitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.delitem(z, 1)
    return z


def dict_getitem_int(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return operator.getitem(z, 1)


def dict_getitem_tensor(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return operator.getitem(z, 2)


def dict_setitem_int(x: int, y: paddle.Tensor):
    z = {'x': x, 'y': y}
    operator.setitem(z, 'x', 2)
    return z


def dict_setitem_tensor(x: int, y: paddle.Tensor):
    z = {'x': x, 'y': y}
    operator.setitem(z, 'y', paddle.to_tensor(3))
    return z


def dict_delitem_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    operator.delitem(z, 1)
    return z


def dict_delitem_tensor(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    operator.delitem(z, 2)
    return z


def tuple_getitem_int(x: int, y: paddle.Tensor):
    x = (x, y)
    return operator.getitem(x, 0)


def tuple_getitem_tensor(x: int, y: paddle.Tensor):
    x = (x, y)
    return operator.getitem(x, 1)


def tuple_getitem_slice(x: int, y: paddle.Tensor):
    x = (x, y, 1)
    return operator.getitem(x, slice(0, 2))


def operator_add(x: int, y: paddle.Tensor):
    return operator.add(x, y)


def operator_mul(x: int, y: paddle.Tensor):
    return operator.mul(x, y)


def operator_truth(y: paddle.Tensor):
    return operator.truth(y)


def operator_is_(x: paddle.Tensor, y: paddle.Tensor):
    return (operator.is_(x, x), operator.is_(x, y))


def operator_in_(x: int, y: list):
    return x in y


def operator_not_in_(x: int, y: list):
    return x not in y


def operator_is_not(x: paddle.Tensor, y: paddle.Tensor):
    return (operator.is_not(x, x), operator.is_not(x, y))


def operator_pos(y: int):
    return operator.pos(+y)


class TestOperators(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(True)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)
        e = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype='float32')
        f = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        g = paddle.to_tensor(False)

        self.assert_results(unary_positive, 1)
        self.assert_results(unary_negative, a)
        self.assert_results(unary_not, b)
        self.assert_results(unary_invert, b)

        self.assert_results(binary_power, c, d)
        self.assert_results(binary_multiply, c, d)
        self.assert_results(binary_matrix_multiply, e, f)
        self.assert_results(binary_floor_divide, c, d)
        self.assert_results(binary_true_divide, c, d)
        self.assert_results(binary_modulo, c, d)
        self.assert_results(binary_add, c, d)
        self.assert_results(binary_subtract, c, d)
        self.assert_results(binary_lshift, 10, 2)
        self.assert_results(binary_rshift, 10, 1)
        self.assert_results(binary_and, b, g)
        self.assert_results(binary_or, b, g)
        self.assert_results(binary_xor, b, g)

        self.assert_results(inplace_power, c, d)
        self.assert_results(inplace_multiply, c, d)
        self.assert_results(inplace_matrix_multiply, e, f)
        self.assert_results(inplace_floor_divide, c, d)
        self.assert_results(inplace_true_divide, c, d)
        self.assert_results(inplace_modulo, c, d)
        self.assert_results(inplace_add, c, d)
        self.assert_results(inplace_subtract, c, d)
        self.assert_results(inplace_lshift, 10, 2)
        self.assert_results(inplace_rshift, 10, 1)
        self.assert_results(inplace_and, b, g)
        self.assert_results(inplace_or, b, g)
        self.assert_results(inplace_xor, b, g)

    @test_with_faster_guard
    def test_operator_simple(self):
        self.assert_results(operator_add, 1, paddle.to_tensor(2))
        self.assert_results(operator_mul, 1, paddle.to_tensor(2))
        self.assert_results(operator_truth, paddle.to_tensor(2))
        self.assert_results(
            operator_is_, paddle.to_tensor(2), paddle.to_tensor(3)
        )
        self.assert_results(
            operator_is_not, paddle.to_tensor(2), paddle.to_tensor(3)
        )
        self.assert_results(operator_pos, 1)
        self.assert_results(operator_in_, 12, [1, 2, 12])
        self.assert_results(operator_in_, 12, [1, 2, 3])
        self.assert_results(operator_not_in_, 12, [1, 2, 3])
        self.assert_results(operator_not_in_, 12, [1, 2, 3])

    def test_operator_list(self):
        self.assert_results(list_getitem, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem_slice, 1, paddle.to_tensor(2))
        self.assert_results(list_setitem_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            list_setitem_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results(list_delitem_int, 1, paddle.to_tensor(2))
        self.assert_results(list_delitem_tensor, 1, paddle.to_tensor(2))

    def test_operator_dict(self):
        self.assert_results(dict_getitem_int, 1, paddle.to_tensor(2))
        self.assert_results(dict_getitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results(dict_setitem_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            dict_setitem_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results(dict_delitem_int, 1, paddle.to_tensor(2))
        self.assert_results(dict_delitem_tensor, 1, paddle.to_tensor(2))

    def test_operator_tuple(self):
        self.assert_results(tuple_getitem_int, 1, paddle.to_tensor(2))
        self.assert_results(tuple_getitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results(tuple_getitem_slice, 1, paddle.to_tensor(2))


def run_not_eq(x: paddle.Tensor, y: int):
    out = paddle.reshape(x, [1, -1]) != y
    out = out.astype('float32')
    return out


class TestNotEq(TestCaseBase):
    def test_not_eq(self):
        x = paddle.to_tensor([2])
        y = 3
        self.assert_results(run_not_eq, x, y)


if __name__ == "__main__":
    unittest.main()
