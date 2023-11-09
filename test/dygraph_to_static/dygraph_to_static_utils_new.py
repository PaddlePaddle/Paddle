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
import logging
import os
import unittest
from enum import Flag, auto
from functools import wraps

import numpy as np

import paddle
from paddle import set_flags, static
from paddle.base import core
from paddle.jit.api import sot_mode_guard

"""
# Usage:
class MyTest(Dy2StTestBase):
    @set_to_static_mode(
        ToStaticMode.AST | ToStaticMode.SOT
    )
    @set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_EXE | IrMode.PIR_API)
    def test_case1(self):
        raise ValueError("MyTest 1")

    def test_case2(self):
        raise ValueError("MyTest 2")


class MyTest2(MyTest):
    def test_case1(self):
        raise ValueError("MyTest2 1")
"""

logger = logging.getLogger("Dygraph to static utils")
logger.setLevel(logging.DEBUG)


class ToStaticMode(Flag):
    AST = auto()
    SOT = auto()

    def lower_case_name(self):
        return self.name.lower()


class IrMode(Flag):
    LEGACY_IR = auto()
    # pir translator mode, Reference link: https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/IR_Dialect/program_translator.md
    PIR_EXE = auto()
    # using native pir api mode
    PIR_API = auto()

    def lower_case_name(self):
        return self.name.lower()


DEFAULT_TO_STATIC_MODE = ToStaticMode.AST | ToStaticMode.SOT
DEFAULT_IR_MODE = IrMode.PIR_API


def to_legacy_ast_test(fn):
    """
    convert run fall_back to ast
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[AST] running AST")
        with sot_mode_guard(False):
            fn(*args, **kwargs)

    return impl


def to_sot_test(fn):
    """
    convert run fall_back to ast
    """

    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[SOT] running SOT")
        with sot_mode_guard(True):
            fn(*args, **kwargs)

    return impl


def to_legacy_ir_test(fn):
    def impl(*args, **kwargs):
        logger.info("[Program] running legacy ir")
        return fn(*args, **kwargs)

    return impl


def to_pir_exe_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[PIR_EXE] running pir exe")
        ir_outs = None
        if os.environ.get('FLAGS_use_stride_kernel', False):
            return
        with static.scope_guard(static.Scope()):
            with static.program_guard(static.Program()):
                try:
                    pir_flag = 'FLAGS_enable_pir_in_executor'
                    os.environ[pir_flag] = 'True'
                    set_flags({pir_flag: True})
                    ir_outs = fn(*args, **kwargs)
                finally:
                    del os.environ[pir_flag]
                    set_flags({pir_flag: False})
        return ir_outs

    return impl


def to_pir_api_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        logger.info("[PIR_API] running pir api")
        ir_outs = None
        with paddle.pir_utils.IrGuard():
            paddle.disable_static()
            ir_outs = fn(*args, **kwargs)
        return ir_outs

    return impl


# Metaclass and BaseClass
class Dy2StTestMeta(type):
    TO_STATIC_HANDLER_MAP = {
        ToStaticMode.SOT: to_sot_test,
        ToStaticMode.AST: to_legacy_ast_test,
    }

    IR_HANDLER_MAP = {
        IrMode.LEGACY_IR: to_legacy_ir_test,
        IrMode.PIR_EXE: to_pir_exe_test,
        IrMode.PIR_API: to_pir_api_test,
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
                    if attr.startswith(f"{fn_name}__"):
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
                # NOTE(gouzil): Temporarily not supported SOT + PIR, link: https://github.com/PaddlePaddle/Paddle/pull/58630
                # if (
                #     to_static_mode == ToStaticMode.SOT
                #     and ir_mode == IrMode.PIR_API
                # ):
                #     continue
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
def test_ast_only(fn):
    fn = set_to_static_mode(ToStaticMode.AST)(fn)
    return fn


def test_sot_only(fn):
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    return fn


def test_pir_only(fn):
    # fn = set_ir_mode(IrMode.PIR_EXE)(fn)
    return fn


def test_legacy_and_pir(fn):
    # fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_EXE)(fn)
    return fn


def test_legacy_and_pir_api(fn):
    # fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_API)(fn)
    return fn


def test_legacy_and_pir_exe_and_pir_api(fn):
    # fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_API | IrMode.PIR_EXE)(fn)
    return fn


def compare_legacy_with_pir(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        outs = fn(*args, **kwargs)
        if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
            return outs
        ir_outs = to_pir_exe_test(fn)(*args, **kwargs)
        np.testing.assert_equal(
            outs,
            ir_outs,
            err_msg=f'Dy2St Unittest Check ({fn.__name__}) has diff \n'
            + f'Expect {outs}\n'
            + f'But Got {ir_outs}',
        )
        return outs

    return impl


# For debug
def show_all_test_cases(test_class):
    logger.info(f"[showing {test_class.__name__}]")
    for attr in dir(test_class):
        if attr.startswith("test"):
            fn = getattr(test_class, attr)
            logger.info(f"{attr}: {fn}")
