# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import math
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import (
    allow_dynamic_shape_guard,
)


def dynamic_shape_input_func1(x):
    s = x.shape[0]
    return x + s


def dynamic_int_input_func1(x, n):
    x = paddle.reshape(x, [n, -1])
    return (x + n) * 2 - 1, (-n + 1) * 2 - 1, type(n) is int


def dynamic_shape_with_constraints(x, n):
    return (x + n) * 2


def dynamic_int_input_func2(x, n):
    return x + n[1]


def dynamic_int_input_func3(x, n):
    if n < 4:
        return 1
    x = paddle.reshape(x, [n, -1])
    return (x + n) * 2 - 1, (-n + 1) * 2 - 1


def dynamic_shape_access_inner_var_shape(x):
    y = x + 1
    return y.shape[0]


def dynamic_shape_in_list(x, shape):
    return x.reshape(shape)


def dynamic_shape_int_mul_float(x):
    y = x * 0.5
    z = math.sin(y)  # Trigger get_py_value
    return z


class CustomConv(paddle.nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @paddle.jit.to_static(full_graph=False)
    def forward(self, x):
        return paddle.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            [self._stride[0] + 1, self._stride[1]],
            self._padding,
            self._dilation,
            self._groups,
            self._data_format,
        )


def pool2d_fallback(x, kernel_size):
    return paddle.nn.functional.max_pool2d(x, kernel_size=kernel_size)


class TestOpcodeExecutorDynamicShapeCache(TestCaseBase):
    def test_dynamic_int_input_cache_hit_case1(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_int_input_func1, paddle.randn([4, 5, 6]), 2
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_int_input_func1, paddle.randn([4, 5, 6]), i
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_int_input_cache_hit_case2(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_int_input_func2, paddle.randn([4, 5, 6]), {1: 2}
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_int_input_func2, paddle.randn([4, 5, 6]), {1: i}
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_int_input_cache_hit_case3(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(0, 6):
                self.assert_results(
                    dynamic_int_input_func3, paddle.randn([4, 5, 6]), i
                )
                self.assertEqual(ctx.translate_count, i + 1)

    def test_dynamic_shape_input_cache_hit_case1(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_input_func1, paddle.randn([2, 4, 5])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_shape_input_func1, paddle.randn([i, 4, 5])
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_shape_input_cache_hit_case2(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_access_inner_var_shape, paddle.randn([2, 4, 5])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_shape_access_inner_var_shape,
                    paddle.randn([i, 4, 5]),
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_shape_cast(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            func1 = check_no_breakgraph(lambda n: bool(n))
            func2 = check_no_breakgraph(lambda n: int(n))
            func3 = check_no_breakgraph(lambda n: float(n))
            for func in [func1, func2, func3]:
                self.assert_results(func, 1)
                self.assert_results(func, 2)

    def test_dynamic_shape_in_list(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_in_list,
                paddle.randn([2, 2, 5]),
                [4, 5],
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_shape_in_list,
                    paddle.randn([i, 2, 5]),
                    [i * 2, 5],
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_conv_dynamic_shape_stride_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 5):
                conv = CustomConv(3, 3, 3, stride=i)
                conv(paddle.randn([1, 3, 224, 224]))
                self.assertEqual(ctx.translate_count, i)

    def test_conv_dynamic_shape_kernel_size_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 5):
                x = paddle.randn([1, 3, 224, 224])
                self.assert_results(pool2d_fallback, x, i)
                self.assertEqual(ctx.translate_count, i)

    def test_pad_dynamic_shape_fallback(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            pad_func = check_no_breakgraph(
                lambda x, n: paddle.nn.functional.pad(x, [0, n, 0, 0])
            )
            for i in range(1, 5):
                self.assert_results(pad_func, paddle.randn([1, 3, 224, 224]), i)
                self.assertEqual(ctx.translate_count, i)

    def test_dynamic_shape_int_mul_float(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            for i in range(1, 6):
                self.assert_results(dynamic_shape_int_mul_float, i)

    def test_mixed_dynamic_and_static(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            a = paddle.randn([4, 5, 6])
            self.assert_results(dynamic_int_input_func1, a, 1)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(dynamic_int_input_func1, a, 0)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(dynamic_int_input_func1, a, 2)
            self.assertEqual(ctx.translate_count, 3)
            for i in range(3, 6):
                self.assert_results(dynamic_int_input_func1, a, i)
                self.assertEqual(ctx.translate_count, 4)

    def test_mixed_static_after_dynamic(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            a = paddle.randn([4, 5, 6])
            self.assert_results(dynamic_int_input_func1, a, 2)
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 6):
                self.assert_results(dynamic_int_input_func1, a, i)
                self.assertEqual(ctx.translate_count, 2)
            self.assert_results(dynamic_int_input_func1, a, 0)
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(dynamic_int_input_func1, a, 1)
            self.assertEqual(ctx.translate_count, 4)

    def test_dynamic_shape_with_constraints(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_with_constraints, paddle.randn([4, 5, 6]), 2
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(3, 7):
                self.assert_results(
                    dynamic_shape_with_constraints,
                    paddle.randn([4 + i, 5, 6]),
                    i,
                )
                self.assertEqual(ctx.translate_count, 2)


@check_no_breakgraph
def dynamic_shape_non_break_non_inplace_ops(x):
    s0 = x.shape[0]
    s1 = s0 + 1
    s2 = 1 + s0
    s3 = s1 + s2
    s4 = s1 * s2
    s5 = s1 - s2
    s6 = s1 / s2
    s7 = s1 // s2
    s8 = s1 % s2
    s9 = s1**s2
    s10 = s1 & s2
    s11 = s1 | s2
    s12 = s1 ^ s2
    s13 = s1 << s2
    s14 = s1 >> s2
    s15 = s1 == s2
    s16 = s1 != s2
    s17 = s1 < s2
    s18 = s1 <= s2
    s19 = s1 > s2
    s20 = s1 >= s2
    s21 = bool(s1)
    s22 = not s2
    return (
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
        s17,
        s18,
        s19,
        s20,
        s21,
        s22,
    )


@check_no_breakgraph
def dynamic_shape_non_break_inplace_ops(x):
    s0 = x.shape[0]
    s1 = s0 + 1
    s2 = s3 = s4 = s5 = s6 = s7 = s8 = s9 = s10 = s11 = s12 = s13 = s0
    s2 += s1
    s3 *= s1
    s4 -= s1
    # TODO(SigureMo): Open this case, currently the compute result between Python and C++ (Paddle Kernel)
    # has a small difference (0.8333333134651184 and 0.8333333333333334)
    # s5 /= s1
    s6 //= s1
    s7 %= s1
    s8 **= s1
    s9 &= s1
    s10 |= s1
    s11 ^= s1
    s12 <<= s1
    s13 >>= s1
    return (
        s0,
        s1,
        s2,
        s3,
        s4,
        # s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
    )


class TestDynamicShapeNonBreakOps(TestCaseBase):
    def test_dynamic_shape_non_break_non_inplace_ops(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_non_break_non_inplace_ops, paddle.randn([4, 5, 6])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(5, 9):
                self.assert_results(
                    dynamic_shape_non_break_non_inplace_ops,
                    paddle.randn([i, 5, 6]),
                )
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_shape_non_break_inplace_ops(self):
        with allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_shape_non_break_inplace_ops, paddle.randn([4, 5, 6])
            )
            self.assertEqual(ctx.translate_count, 1)
            for i in range(5, 9):
                self.assert_results(
                    dynamic_shape_non_break_inplace_ops,
                    paddle.randn([i, 5, 6]),
                )
                self.assertEqual(ctx.translate_count, 2)


if __name__ == '__main__':
    unittest.main()
