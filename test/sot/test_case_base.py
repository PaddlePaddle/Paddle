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
import inspect
import os
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


def github_action_error_msg(msg: str):
    if 'GITHUB_ACTIONS' in os.environ:
        frame = inspect.currentframe()
        if frame is not None:
            # find the first frame that is in the test folder
            while frame.f_back is not None:
                filename = frame.f_code.co_filename
                if filename.startswith("./"):
                    filename = f"tests/{filename[2:]}"
                    lineno = frame.f_lineno
                    output = f"\n::error file={filename},line={lineno}::{msg}"
                    return output
                frame = frame.f_back
    return None


class TestCaseBase(unittest.TestCase):
    def assertIs(self, x, y, msg=None):
        super().assertIs(x, y, msg=msg)
        if msg is None:
            msg = f"Assert Is, x is {x}, y is {y}"
        msg = github_action_error_msg(msg)
        if msg is not None:
            print(msg)

    def assertEqual(self, x, y, msg=None):
        super().assertEqual(x, y, msg=msg)
        if msg is None:
            msg = f"Assert Equal, x is {x}, y is {y}"
        msg = github_action_error_msg(msg)
        if msg is not None:
            print(msg)

    def assert_nest_match(self, x, y):
        cls_x = type(x)
        cls_y = type(y)
        msg = f"type mismatch, x is {cls_x}, y is {cls_y}"
        self.assertIs(cls_x, cls_y, msg=msg)

        container_types = (tuple, list, dict, set)
        if cls_x in container_types:
            msg = f"length mismatch, x is {len(x)}, y is {len(y)}"
            self.assertEqual(
                len(x),
                len(y),
                msg=msg,
            )
            if cls_x in (tuple, list):
                for x_item, y_item in zip(x, y):
                    self.assert_nest_match(x_item, y_item)
            elif cls_x is dict:
                for x_key, y_key in zip(x.keys(), y.keys()):
                    self.assert_nest_match(x_key, y_key)
                    self.assert_nest_match(x[x_key], y[y_key])
            elif cls_x is set:
                # TODO: Nested set is not supported yet
                self.assertEqual(x, y)
        elif cls_x in (np.ndarray, paddle.Tensor):
            # TODO: support assert_allclose github error log
            np.testing.assert_allclose(x, y)
        else:
            self.assertEqual(x, y)

    def assert_results(self, func, *inputs):
        sym_output = symbolic_translate(func)(*inputs)
        paddle_output = func(*inputs)
        self.assert_nest_match(sym_output, paddle_output)

    def assert_results_with_side_effects(self, func, *inputs):
        sym_inputs = copy.deepcopy(inputs)
        sym_output = symbolic_translate(func)(*sym_inputs)
        paddle_inputs = copy.deepcopy(inputs)
        paddle_output = func(*paddle_inputs)
        self.assert_nest_match(sym_inputs, paddle_inputs)
        self.assert_nest_match(sym_output, paddle_output)

    def assert_results_with_global_check(
        self, func, global_keys: list[str], *inputs
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
        sym_output = sym_fn(*inputs)
        paddle_output = paddle_fn(*inputs)
        for key in global_keys:
            self.assert_nest_match(
                sym_copied_fn.__globals__[key], paddle_fn.__globals__[key]
            )
        self.assert_nest_match(sym_output, paddle_output)
