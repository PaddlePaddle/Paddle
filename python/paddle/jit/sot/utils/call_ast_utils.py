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

import inspect
import types

import paddle

from .envs import ENV_SOT_WITH_CONTROL_FLOW
from .exceptions import InnerError
from .utils import Singleton

try_ast_codes = set()


def try_ast_func(func):
    def _is_wrapped(f):
        return hasattr(f, '__wrapped__')

    unwrapped_f = func
    if hasattr(unwrapped_f, "__code__"):
        try_ast_codes.add(func.__code__)

    while _is_wrapped(unwrapped_f):
        unwrapped_f = unwrapped_f.__wrapped__
        if hasattr(unwrapped_f, "__code__"):
            try_ast_codes.add(func.__code__)

    return func


class StaticFunctionManager(metaclass=Singleton):
    def __init__(self):
        self.code_map = {}

    def ast_transform_with_frame(self, frame):
        code = frame.f_code
        if code not in try_ast_codes:
            return None
        if code not in self.code_map:
            if code.co_name.startswith("#") or code.co_name.startswith("$"):
                self.code_map[code] = None
            elif len(code.co_cellvars) + len(code.co_freevars) != 0:
                self.code_map[code] = None
            else:
                function = types.FunctionType(
                    code,
                    frame.f_globals,
                    code.co_name,
                    (),
                    (),
                )
                function = paddle.jit.to_static(function, full_graph=True)
                self.code_map[code] = function

        return self.code_map[code]

    def ast_transform_with_callable(self, fn):
        if not inspect.isfunction(fn) or not hasattr(fn, "__code__"):
            return None

        code = fn.__code__
        if code not in try_ast_codes:
            return None
        if code not in self.code_map:
            if code.co_name.startswith("#") or code.co_name.startswith("$"):
                self.code_map[code] = None
            elif len(code.co_cellvars) + len(code.co_freevars) != 0:
                self.code_map[code] = None
            else:
                self.code_map[code] = paddle.jit.to_static(fn, full_graph=True)

        return self.code_map[code]


def get_static_function(obj, type_):
    if ENV_SOT_WITH_CONTROL_FLOW.get():
        if type_ == "eval_frame":
            return StaticFunctionManager().ast_transform_with_frame(obj)
        elif type_ == "inline_call":
            return StaticFunctionManager().ast_transform_with_callable(obj)
        else:
            raise InnerError(f"Can not get static function with type {type_}.")
    return None
