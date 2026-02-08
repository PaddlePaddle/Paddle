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

from __future__ import annotations

import contextlib
import copy
import types
import unittest

import numpy as np

import paddle
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)


@contextlib.contextmanager
def test_instruction_translator_cache_context():
    cache = OpcodeExecutorCache()
    cache.clear()
    yield cache
    cache.clear()


class TestCaseBase(unittest.TestCase):
    def assert_nest_match(self, actual, expected):
        cls_actual = type(actual)
        cls_expected = type(expected)
        msg = (
            f"type mismatch, actual is {cls_actual}, expected is {cls_expected}"
        )
        self.assertIs(cls_actual, cls_expected, msg=msg)

        container_types = (tuple, list, dict, set)
        if cls_actual in container_types:
            msg = f"length mismatch, actual is {len(actual)}, expected is {len(expected)}"
            self.assertEqual(
                len(actual),
                len(expected),
                msg=msg,
            )
            if cls_actual in (tuple, list):
                for actual_item, expected_item in zip(actual, expected):
                    self.assert_nest_match(actual_item, expected_item)
            elif cls_actual is dict:
                for actual_key, expected_key in zip(
                    actual.keys(), expected.keys()
                ):
                    self.assert_nest_match(actual_key, expected_key)
                    self.assert_nest_match(
                        actual[actual_key], expected[expected_key]
                    )
            elif cls_actual is set:
                # TODO: Nested set is not supported yet
                self.assertEqual(actual, expected)
        elif cls_actual in (np.ndarray, paddle.Tensor):
            # TODO: support assert_allclose github error log
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
        else:
            self.assertEqual(actual, expected)

    def assert_results(self, func, *args, **kwargs):
        sym_output = symbolic_translate(func)(*args, **kwargs)
        paddle_output = func(*args, **kwargs)
        self.assert_nest_match(sym_output, paddle_output)

    def assert_results_with_grad(self, inputs, func, *args, **kwargs):
        def _find_all_tensors(obj):
            ret = []
            container_types = (tuple, list, set)
            if isinstance(obj, container_types):
                for item in obj:
                    ret.extend(_find_all_tensors(item))
            elif isinstance(obj, dict):
                for value in obj.values():
                    ret.extend(_find_all_tensors(value))
            elif isinstance(obj, paddle.Tensor):
                ret.append(obj)
            return ret

        def _accumulate(tensors: list):
            out = paddle.empty(shape=[], dtype='float64')
            for tensor in tensors:
                out += paddle.mean(tensor.astype('float64'))
            return out

        def _cal_input_grads(outputs):
            tensor_outs = _find_all_tensors(outputs)
            acc = _accumulate(tensor_outs)
            acc.backward()
            tensor_inputs = _find_all_tensors(inputs)
            input_grads = []
            for input in tensor_inputs:
                input_grads.append(
                    None if input.grad is None else input.grad.clone()
                )
                input.clear_gradient()
            return input_grads

        sym_output = symbolic_translate(func)(*args, **kwargs)
        paddle_output = func(*args, **kwargs)
        sym_input_grads = _cal_input_grads(sym_output)
        paddle_input_grads = _cal_input_grads(paddle_output)
        self.assert_nest_match(sym_input_grads, paddle_input_grads)
        self.assert_nest_match(sym_output, paddle_output)

    def assert_exceptions(self, exec, info, func, *args, **kwargs):
        self.assertRaisesRegex(
            exec, info, symbolic_translate(func), *args, **kwargs
        )

    def assert_results_with_side_effects(self, func, *args, **kwargs):
        sym_args, sym_kwargs = copy.deepcopy((args, kwargs))
        sym_output = symbolic_translate(func)(*sym_args, **sym_kwargs)
        paddle_args, paddle_kwargs = copy.deepcopy((args, kwargs))
        paddle_output = func(*paddle_args, **paddle_kwargs)
        self.assert_nest_match(sym_args, paddle_args)
        self.assert_nest_match(sym_kwargs, paddle_kwargs)
        self.assert_nest_match(sym_output, paddle_output)

    def assert_results_with_global_check(
        self, func, global_keys: list[str], *args, **kwargs
    ):
        def copy_fn(fn):
            return types.FunctionType(
                code=fn.__code__,
                globals=copy.copy(fn.__globals__),
                name=fn.__name__,
                argdefs=fn.__defaults__,
                closure=fn.__closure__,
            )

        sym_copied_fn = copy_fn(func)
        sym_fn = symbolic_translate(sym_copied_fn)
        paddle_fn = copy_fn(func)
        sym_output = sym_fn(*args, **kwargs)
        paddle_output = paddle_fn(*args, **kwargs)
        for key in global_keys:
            self.assert_nest_match(
                sym_copied_fn.__globals__[key], paddle_fn.__globals__[key]
            )
        self.assert_nest_match(sym_output, paddle_output)
