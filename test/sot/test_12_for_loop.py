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

# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
    test_with_faster_guard,
)

import paddle
from paddle.jit import sot
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.utils import strict_mode_guard


def gener():
    yield 1
    yield 2
    yield 3


def for_list_1(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if x > 2:
            x += 1
        else:
            x -= 1
    return x


def for_list_2(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if i > 2:
            x += 1
        else:
            x -= 1
    return x


def for_dict(x: paddle.Tensor):
    map = {1: 2, 3: 4}
    for k in map.keys():
        x += k

    for v in map.values():
        x += v

    for k, v in map.items():
        x += k
        x += v

    return x


def for_iter(x, it):
    for item in it:
        x += item
    return x


def for_for_fallback(x, it):
    for i in [1, 2, 3]:
        for item in it:
            x += item
    return x


def for_break(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        x += i
        if i == 2:
            break
    for i in it:
        x += i
        if i == 2:
            break
    return x


def for_continue(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        if i == 2:
            continue
        x += i

    for i in it:
        if i == 2:
            continue
        x += i
    return x


def for_enumerate_var_with_nested_range(x_array):
    x = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = paddle.to_tensor(x_array)
    for i, num in enumerate(x_array):
        for idx in range(num):
            x = x + num
    return x


def for_create_tmp_in_loop(x, it):
    s = x
    for i in it:
        tmp = i
        s += tmp
    return s, tmp


def for_without_zero_iter(self_res_dict, output):
    res_dict = {"logits": output}
    for res_key in list(self_res_dict):
        res_dict[res_key] = self_res_dict.pop(res_key)
    return res_dict


@sot.psdb.check_no_fallback
def for_reconstruct_range_iter():
    for i in range(3):
        sot.psdb.breakgraph()


global_var_name = None


def for_tmp_var_with_same_name_as_global_var():
    total = 0
    for i in range(3):
        global_var_name = i + 3
        sot.psdb.breakgraph()
        total += global_var_name
    return total


def for_layer_list(layer_list, x):
    for net in layer_list:
        x = net(x)
    return x


class TestForLoop(TestCaseBase):
    def test_list(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_1, a)

    @test_with_faster_guard
    def test_list_with_fallback(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_2, a)

    def test_dict(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_dict, a)

    def test_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_for_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    @strict_mode_guard(False)
    def test_for_break(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_break)(a, gener())
        paddle_output = for_break(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    @test_with_faster_guard
    def test_for_continue(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_continue)(a, gener())
        paddle_output = for_continue(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    # TODO(zmh): support range for tensor
    # def test_resume_stack(self):
    #     a = [1, 2, 3]
    #     self.assert_results(for_enumerate_var_with_nested_range, a)

    @test_with_faster_guard
    def test_create_var_in_loop(self):
        x = paddle.to_tensor(1, dtype="float32")
        a = [1, 2, 3]
        self.assert_results(for_create_tmp_in_loop, x, a)

        sym_output = symbolic_translate(for_create_tmp_in_loop)(x, iter(a))
        paddle_output = for_create_tmp_in_loop(x, iter(a))
        self.assert_nest_match(sym_output, paddle_output)

    def test_create_var_in_loop_with_same_name_as_global(self):
        self.assert_results(for_tmp_var_with_same_name_as_global_var)

    @test_with_faster_guard
    def test_for_without_zero_iter(self):
        self_res_dict = {}
        output = paddle.to_tensor(2)
        self.assert_results(for_without_zero_iter, self_res_dict, output)

    def test_reconstruct_range_iter(self):
        self.assert_results(for_reconstruct_range_iter)

    @test_with_faster_guard
    def test_layer_list(self):
        layers = paddle.nn.LayerList()
        for i in range(5):
            layers.append(paddle.nn.Linear(5, 5))
        x = paddle.rand([5], dtype="float32")
        self.assert_results(for_layer_list, layers, x)


def run_list_comp(x):
    out = [s.chunk(2, axis=1) for s in x]
    return out


class TestListComp(TestCaseBase):
    def test_list_comp(self):
        x = [paddle.randn([1, 4]), paddle.randn([1, 4])]
        self.assert_results(run_list_comp, x)


def for_enumerate_cache(func_list, x):
    out = None
    for idx, func in enumerate(func_list):
        out = func(x[idx])
    return out


class TestEnumerateCache(TestCaseBase):
    @test_with_faster_guard
    def test_run(self):
        func_list = [
            paddle.nn.Linear(10, 10),
        ]
        x = [
            paddle.randn([5, 10]),
        ]

        with test_instruction_translator_cache_context() as ctx:
            out = symbolic_translate(for_enumerate_cache)(func_list, x)
            out = symbolic_translate(for_enumerate_cache)(func_list, x)
            self.assertEqual(ctx.translate_count, 1)


# after_loop_fn need zzz, and zzz is created as UndefinedVar when generating loop body
# do not set zzz as UndefinedVar again
def undefined_var_case_0():
    for i in [1, 2]:
        sot.psdb.breakgraph()
        zzz = i

    zzz = zzz + 1
    return zzz


# after_loop_fn need create zzz as UndefinedVar
def undefined_var_case_1():
    for i in [1, 2]:
        sot.psdb.breakgraph()
        aaa = i

    for i in [1, 3]:
        zzz = i
    zzz = zzz + 1
    return zzz


class TestUndefinedVarInRiskyCodes(TestCaseBase):
    def test_undefined_var_case_0(self):
        self.assert_results(undefined_var_case_0)

    @test_with_faster_guard
    def test_undefined_var_case_1(self):
        self.assert_results(undefined_var_case_1)


def comp_with_fallback(x):
    paddle.jit.sot.psdb.fallback()
    y = [len(t) for t in x]
    return paddle.to_tensor(y)


class TestListCompWithFallback(TestCaseBase):
    @strict_mode_guard(False)
    def test_list_comp_with_fallback(self):
        x = [paddle.randn([4, 6])]
        self.assert_results(comp_with_fallback, x)


def for_arange(x):
    for i in paddle.arange(0, 5):
        x = x + i
    return x


class TestArange(TestCaseBase):
    def test_arange(self):
        x = paddle.to_tensor(1)
        self.assert_results(for_arange, x)


def for_break_with_load_same_consts(x: paddle.Tensor):
    y = None
    z = None
    for i in [1, 2, 3]:
        if y is None:
            y = i
        if z is None:
            z = i
        x += y + z
        sot.psdb.breakgraph()
    return x


class TestForBreakWithLoadSameConsts(TestCaseBase):
    def test_for_break_with_load_same_consts(self):
        x = paddle.to_tensor(1)
        self.assert_results(for_break_with_load_same_consts, x)


def for_break_with_write_pre_defined_name(x: paddle.Tensor):
    y = None
    for i in [1, 2, 3]:
        y = i
        sot.psdb.breakgraph()
    return x + 1


class TestForBreakWithWritePreDefinedName(TestCaseBase):
    def test_for_break_with_write_pre_defined_name(self):
        x = paddle.to_tensor(1)
        self.assert_results(for_break_with_write_pre_defined_name, x)


if __name__ == "__main__":
    unittest.main()
