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
import os
from functools import wraps

import numpy as np

from paddle import set_flags, static
from paddle.base import core


@contextlib.contextmanager
def enable_fallback_guard(enable):
    flag = os.environ.get("ENABLE_FALL_BACK", None)
    os.environ["ENABLE_FALL_BACK"] = enable
    yield
    if flag is not None:
        os.environ["ENABLE_FALL_BACK"] = flag
    else:
        del os.environ["ENABLE_FALL_BACK"]


def to_ast(func):
    """
    convert run fall_back to ast
    """

    def impl(*args, **kwargs):
        with enable_fallback_guard("False"):
            func(*args, **kwargs)

    return impl


def to_sot(func):
    """
    convert run fall_back to ast
    """
    enable_sot = os.environ.get("ENABLE_SOT", "False") == "True"

    def impl(*args, **kwargs):
        if enable_sot:
            with enable_fallback_guard("True"):
                func(*args, **kwargs)
        else:
            return

    return impl


def dy2static_unittest(cls):
    """
    dy2static unittest must be decorated to each Dy2static Unittests.
    run both in Fallback and Ast mode.

    Examples:

        >>> @dy2static_unittest
        ... class TestA(unittest.TestCase):
        ...     ...
    """
    for key in dir(cls):
        if key.startswith("test"):
            if not key.endswith("_ast"):
                test_func = getattr(cls, key)
                setattr(cls, key + "_ast", to_ast(test_func))
            test_func = getattr(cls, key)
            setattr(cls, key, to_sot(test_func))
    return cls


def ast_only_test(func):
    """
    run this test function in ast only mode.

    Examples:

        >>> @dy2static_unittest
        ... class TestA(unittest.TestCase):
        ...     @ast_only_test
        ...     def test_ast_only(self):
        ...         pass
    """

    def impl(*args, **kwargs):
        if os.environ.get("ENABLE_FALL_BACK", "False") == "False":
            func(*args, **kwargs)

    return impl


def sot_only_test(func):
    """
    run this test function in ast only mode.

    Examples:

        >>> @dy2static_unittest
        ... class TestA(unittest.TestCase):
        ...     @sot_only_test
        ...     def test_sot_only(self):
        ...         pass
    """

    def impl(*args, **kwargs):
        if os.environ.get("ENABLE_FALL_BACK", "False") == "True":
            func(*args, **kwargs)

    return impl


def test_with_new_ir(func):
    @wraps(func)
    def impl(*args, **kwargs):
        ir_outs = None
        if os.environ.get('FLAGS_use_stride_kernel', False):
            return
        with static.scope_guard(static.Scope()):
            with static.program_guard(static.Program()):
                try:
                    new_ir_flag = 'FLAGS_enable_new_ir_in_executor'
                    os.environ[new_ir_flag] = 'True'
                    set_flags({new_ir_flag: True})
                    ir_outs = func(*args, **kwargs)
                finally:
                    del os.environ[new_ir_flag]
                    set_flags({new_ir_flag: False})
        return ir_outs

    return impl


def test_and_compare_with_new_ir(need_check_output: bool = True):
    def decorator(func):
        @wraps(func)
        def impl(*args, **kwargs):
            outs = func(*args, **kwargs)
            if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
                return outs
            ir_outs = test_with_new_ir(func)(*args, **kwargs)
            if not need_check_output:
                return outs
            np.testing.assert_equal(
                outs,
                ir_outs,
                err_msg='Dy2St Unittest Check ('
                + func.__name__
                + ') has diff '
                + '\nExpect '
                + str(outs)
                + '\n'
                + 'But Got'
                + str(ir_outs),
            )
            return outs

        return impl

    return decorator
