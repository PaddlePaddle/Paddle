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
import logging
import os
import unittest
from enum import Flag, auto
from functools import wraps

import numpy as np

from paddle import set_flags, static
from paddle.base import core

"""
# Usage:
class MyTest(Dy2StTestBase):
    @set_to_static_mode(
        ToStaticMode.LEGACY_AST | ToStaticMode.SOT | ToStaticMode.PIR_AST
    )
    @set_ir_mode(IrMode.LEGACY_PROGRAM | IrMode.PIR)
    def test_case1(self):
        raise ValueError("MyTest 1")

    def test_case2(self):
        raise ValueError("MyTest 2")


class MyTest2(MyTest):
    def test_case1(self):
        raise ValueError("MyTest2 1")
"""

logger = logging.getLogger("Dygraph to static utils")
logger.setLevel(logging.WARNING)


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


def in_sot_mode():
    return os.getenv("ENABLE_FALL_BACK", "False") == "True"


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
        logger.info("[AST] running AST")
        with enable_fallback_guard("False"):
            fn(*args, **kwargs)

    return impl


def to_sot_test(fn):
    """
    convert run fall_back to ast
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[SOT] running SOT")
        with enable_fallback_guard("True"):
            fn(*args, **kwargs)

    return impl


def to_pir_ast_test(fn):
    raise TypeError("Don't enable PIR AST mode now!")


def to_legacy_program_test(fn):
    def impl(*args, **kwargs):
        logger.info("[Program] running legacy program")
        return fn(*args, **kwargs)

    return impl


def to_pir_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[PIR] running pir")
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
            if key.startswith("test") and inspect.isfunction(value)
        }
        logger.info(f"[creating {name}]")
        new_attrs.update(
            {
                key: value
                for key, value in attrs.items()
                if key not in original_test_cases
            }
        )
        for fn_name, fn in original_test_cases.items():
            logger.info(f"Generating {fn_name}")
            # Disable inherited test cases
            for base in bases:
                for attr in dir(base):
                    if attr.startswith(fn_name):
                        new_attrs[attr] = None
            fn_to_static_modes = getattr(
                fn, "to_static_mode", DEFAULT_TO_STATIC_MODE
            )
            fn_ir_modes = getattr(fn, "ir_mode", DEFAULT_IR_MODE)
            fn_disabled_test_cases = getattr(fn, "disabled_test_cases", [])
            logger.info(f"fn_to_static_modes: {fn_to_static_modes}")
            logger.info(f"fn_ir_modes: {fn_ir_modes}")
            logger.info(f"fn_disabled_test_cases: {fn_disabled_test_cases}")
            # Get all valid test cases with to_static_mode and ir_mode
            to_static_with_ir_modes = [
                (to_static_mode, ir_mode)
                for to_static_mode in ToStaticMode
                for ir_mode in IrMode
                if to_static_mode & fn_to_static_modes and ir_mode & fn_ir_modes
            ]
            # Filter out disabled test cases and test cases already in compare groups
            to_static_with_ir_modes = list(
                filter(
                    lambda flags: (flags not in fn_disabled_test_cases),
                    to_static_with_ir_modes,
                )
            )
            # Generate all test cases
            for to_static_mode, ir_mode in to_static_with_ir_modes:
                if (
                    to_static_mode == ToStaticMode.PIR_AST
                    and ir_mode == IrMode.LEGACY_PROGRAM
                ):
                    # PIR with LEGACY_PROGRAM is not a valid combination
                    continue
                new_attrs[
                    Dy2StTestMeta.test_case_name(
                        fn_name, to_static_mode, ir_mode
                    )
                ] = Dy2StTestMeta.convert_test_case(fn, to_static_mode, ir_mode)
        return type.__new__(cls, name, bases, new_attrs)

    @staticmethod
    def test_case_name(original_name: str, to_static_mode, ir_mode):
        return f"{original_name}__{to_static_mode.lower_case_name()}_{ir_mode.lower_case_name()}"

    @staticmethod
    def convert_test_case(fn, to_static_mode, ir_mode):
        fn = Dy2StTestMeta.IR_HANDLER_MAP[ir_mode](fn)
        fn = Dy2StTestMeta.TO_STATIC_HANDLER_MAP[to_static_mode](fn)
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


def _test_and_compare_with_new_ir(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        outs = fn(*args, **kwargs)
        if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
            return outs
        ir_outs = to_pir_test(fn)(*args, **kwargs)
        np.testing.assert_equal(
            outs,
            ir_outs,
            err_msg=f'Dy2St Unittest Check ({fn.__name__}) has diff \n'
            + f'Expect {outs}\n'
            + f'But Got {ir_outs}',
        )
        return outs

    return impl


def test_and_compare_with_new_ir(need_check_output: bool = True):
    def decorator(fn):
        fn = set_ir_mode(IrMode.LEGACY_PROGRAM | IrMode.PIR)(fn)
        if need_check_output:
            logger.info(f"[need_check_output] {fn.__name__}")
            fn = _test_and_compare_with_new_ir(fn)
        return fn

    return decorator


# For debug
def show_all_test_cases(test_class):
    logger.info(f"[showing {test_class.__name__}]")
    for attr in dir(test_class):
        if attr.startswith("test"):
            fn = getattr(test_class, attr)
            logger.info(f"{attr}: {fn}")
