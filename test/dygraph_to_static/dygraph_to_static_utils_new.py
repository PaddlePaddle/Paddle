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

import contextlib
import inspect
import operator
import os
import unittest
from enum import Flag, auto
from functools import reduce, wraps

import numpy as np

from paddle import set_flags, static


class ToStaticMode(Flag):
    LEGACY_AST = auto()
    PIR_AST = auto()
    SOT = auto()

    def lower_case_name(self):
        return self.name.lower()


class IrMode(Flag):
    LEGACY_PROGRAM = auto()
    PIR = auto()

    def lower_case_name(self):
        return self.name.lower()


DEFAULT_TO_STATIC_MODE = ToStaticMode.LEGACY_AST | ToStaticMode.SOT
DEFAULT_IR_MODE = IrMode.LEGACY_PROGRAM


@contextlib.contextmanager
def enable_fallback_guard(enable):
    flag = os.environ.get("ENABLE_FALL_BACK", None)
    os.environ["ENABLE_FALL_BACK"] = enable
    yield
    if flag is not None:
        os.environ["ENABLE_FALL_BACK"] = flag
    else:
        del os.environ["ENABLE_FALL_BACK"]


def to_legacy_ast_test(fn):
    """
    convert run fall_back to ast
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        print("[AST] running AST")
        with enable_fallback_guard("False"):
            fn(*args, **kwargs)

    return impl


def to_sot_test(fn):
    """
    convert run fall_back to ast
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        print("[SOT] running SOT")
        with enable_fallback_guard("True"):
            fn(*args, **kwargs)

    return impl


def to_pir_ast_test(fn):
    raise TypeError("Don't enable PIR AST mode now!")


def to_legacy_program_test(fn):
    def impl(*args, **kwargs):
        print("[Program] running legacy program")
        return fn(*args, **kwargs)

    return impl


def to_pir_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        print("[PIR] running pir")
        ir_outs = None
        if os.environ.get('FLAGS_use_stride_kernel', False):
            return
        with static.scope_guard(static.Scope()):
            with static.program_guard(static.Program()):
                try:
                    new_ir_flag = 'FLAGS_enable_new_ir_in_executor'
                    os.environ[new_ir_flag] = 'True'
                    set_flags({new_ir_flag: True})
                    ir_outs = fn(*args, **kwargs)
                finally:
                    del os.environ[new_ir_flag]
                    set_flags({new_ir_flag: False})
        return ir_outs

    return impl


# Metaclass and BaseClass
class Dy2StTestMeta(type):
    TO_STATIC_HANDLER_MAP = {
        ToStaticMode.SOT: to_sot_test,
        ToStaticMode.LEGACY_AST: to_legacy_ast_test,
        ToStaticMode.PIR_AST: to_pir_ast_test,
    }

    IR_HANDLER_MAP = {
        IrMode.LEGACY_PROGRAM: to_legacy_program_test,
        IrMode.PIR: to_pir_test,
    }

    def __new__(cls, name, bases, attrs):
        new_attrs = {}
        original_test_cases = {
            key: value
            for key, value in attrs.items()
            if key.startswith("test_") and inspect.isfunction(value)
        }
        print(f"[creating {name}]")
        print(attrs)
        new_attrs.update(
            {
                key: value
                for key, value in attrs.items()
                if key not in original_test_cases
            }
        )
        for key, value in original_test_cases.items():
            # Disable inherited test cases
            for base in bases:
                for attr in dir(base):
                    if attr.startswith(key):
                        new_attrs[attr] = None
            fn_to_static_modes = getattr(
                value, "to_static_mode", DEFAULT_TO_STATIC_MODE
            )
            fn_ir_modes = getattr(value, "ir_mode", DEFAULT_IR_MODE)
            fn_compare_groups = getattr(value, "compare_group", [])
            fn_disabled_test_cases = getattr(value, "disabled_test_cases", [])
            print("fn_to_static_modes", fn_to_static_modes)
            print("fn_ir_modes", fn_ir_modes)
            print("fn_disabled_test_cases", fn_disabled_test_cases)
            # Get all valid test cases with to_static_mode and ir_mode
            to_static_with_ir_modes = [
                (to_static_mode, ir_mode)
                for to_static_mode in ToStaticMode
                for ir_mode in IrMode
                if to_static_mode & fn_to_static_modes and ir_mode & fn_ir_modes
            ]
            # Add compare groups and patch it to TestCaseBase
            for compare_group in fn_compare_groups:
                group_name = f"{key}_COMPARE_GROUP"
                for to_static_mode, ir_mode in compare_group:
                    if (to_static_mode, ir_mode) not in to_static_with_ir_modes:
                        raise ValueError(
                            f"Invalid compare group: {compare_group}, please check your test case!"
                        )
                    group_name = Dy2StTestMeta.test_case_name(
                        group_name, to_static_mode, ir_mode
                    )
                new_attrs[group_name] = Dy2StTestMeta.combine_test_cases(
                    value, compare_group, group_name
                )
            flattened_fn_compare_groups = list(
                reduce(operator.add, fn_compare_groups, ())
            )
            print(
                "fn_compare_groups:",
                fn_compare_groups,
                flattened_fn_compare_groups,
            )
            # Filter out disabled test cases and test cases already in compare groups
            to_static_with_ir_modes = list(
                filter(
                    lambda flags: (flags not in fn_disabled_test_cases)
                    and (flags not in flattened_fn_compare_groups),
                    to_static_with_ir_modes,
                )
            )
            # Patch all test cases
            for to_static_mode, ir_mode in to_static_with_ir_modes:
                if (
                    to_static_mode == ToStaticMode.PIR_AST
                    and ir_mode == IrMode.LEGACY_PROGRAM
                ):
                    # PIR with LEGACY_PROGRAM is not a valid combination
                    continue
                new_attrs[
                    Dy2StTestMeta.test_case_name(key, to_static_mode, ir_mode)
                ] = Dy2StTestMeta.convert_test_case(
                    value, to_static_mode, ir_mode
                )
        return type.__new__(cls, name, bases, new_attrs)

    @staticmethod
    def test_case_name(original_name: str, to_static_mode, ir_mode):
        return f"{original_name}__{to_static_mode.lower_case_name()}_{ir_mode.lower_case_name()}"

    @staticmethod
    def combine_test_cases(fn, compare_group, group_name):
        def combined_test_case(self, *args, **kwargs):
            results = []
            # Run each test case in compare group
            print("Running test group: ", group_name)
            for to_static_mode, ir_mode in compare_group:
                print(
                    f"Running test case: to_static_mode is {to_static_mode}, ir_mode is {ir_mode}"
                )
                test_case = Dy2StTestMeta.convert_test_case(
                    fn, to_static_mode, ir_mode
                )
                results.append(test_case(self, *args, **kwargs))
            # Compare each results with first result
            compare_base = results[0]
            for i, result in enumerate(results[1:], 1):
                np.testing.assert_allclose(
                    compare_base,
                    result,
                    err_msg=f"Run compare {compare_group[0]} with {compare_group[i]} failed!",
                )

        combined_test_case.__name__ = group_name
        return combined_test_case

    @staticmethod
    def convert_test_case(fn, to_static_mode, ir_mode):
        fn = Dy2StTestMeta.TO_STATIC_HANDLER_MAP[to_static_mode](fn)
        fn = Dy2StTestMeta.IR_HANDLER_MAP[ir_mode](fn)
        return fn


class Dy2StTestBase(unittest.TestCase, metaclass=Dy2StTestMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Base decorators
def set_to_static_mode(mode: ToStaticMode):
    def decorator(fn):
        fn.to_static_mode = mode
        return fn

    return decorator


def set_ir_mode(mode: IrMode):
    def decorator(fn):
        fn.ir_mode = mode
        return fn

    return decorator


def add_compare_group(*flags):
    def decorator(fn):
        compare_groups = getattr(fn, "compare_group", [])
        compare_groups.append(flags)
        fn.compare_group = compare_groups
        return fn

    return decorator


def disable_test_case(flags):
    def decorator(fn):
        disabled_test_cases = getattr(fn, "disabled_test_cases", [])
        disabled_test_cases.append(flags)
        fn.disabled_test_cases = disabled_test_cases
        return fn

    return decorator


# Suger decorators
# These decorators can be simply composed by base decorators
def ast_only_test(fn):
    fn = set_to_static_mode(ToStaticMode.LEGACY_AST)(fn)
    return fn


def sot_only_test(fn):
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    return fn


def test_with_new_ir(fn):
    fn = set_ir_mode(IrMode.PIR)(fn)
    return fn


def test_and_compare_with_new_ir(compare=False):
    def decorator(fn):
        fn = set_ir_mode(IrMode.LEGACY_PROGRAM | IrMode.PIR)(fn)
        if compare:
            fn = add_compare_group(
                (ToStaticMode.LEGACY_AST, IrMode.LEGACY_PROGRAM),
                (ToStaticMode.LEGACY_AST, IrMode.PIR),
            )(fn)
        return fn

    return decorator


# For debug
def show_all_test_cases(test_class):
    print(f"[{test_class.__name__}]")
    for attr in dir(test_class):
        if attr.startswith("test_"):
            fn = getattr(test_class, attr)
            print(f"{attr}: {fn}")


# class MyTest(Dy2StTestBase):
#     @set_to_static_mode(
#         ToStaticMode.LEGACY_AST | ToStaticMode.SOT | ToStaticMode.PIR_AST
#     )
#     @set_ir_mode(IrMode.LEGACY_PROGRAM | IrMode.PIR)
#     @add_compare_group(
#         (ToStaticMode.LEGACY_AST, IrMode.LEGACY_PROGRAM),
#         (ToStaticMode.LEGACY_AST, IrMode.PIR),
#     )
#     def test_case1(self):
#         print(1)
#         raise ValueError("MyTest 1")

#     def test_case2(self):
#         raise ValueError("MyTest 2")


# class MyTest2(MyTest):
#     def test_case1(self):
#         print(1)
#         raise ValueError("MyTest2 1")
