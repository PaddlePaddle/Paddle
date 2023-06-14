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
    convet run fall_back to ast
    """

    def impl(*args, **kwargs):
        with enable_fallback_guard("False"):
            func(*args, **kwargs)

    return impl


def dy2static_unittest(cls):
    """
    dy2static unittest must be decorated to each Dy2static Unittests.
    run both in Fallback and Ast mode.
    Usage like:

    @dy2static_unittest
    class TestA (unittest.TestCase):
        ...
    """
    for key in dir(cls):
        if key.startswith("test"):
            if not key.endswith("_ast"):
                test_func = getattr(cls, key)
                setattr(cls, key + "_ast", to_ast(test_func))
    return cls


def ast_only_test(func):
    """
    run this test function in ast only mode.
    Usage:

    class TestA (unittest.TestCase):
        @ast_only_test
        def test_ast_only(self):
            pass
    """

    def impl(*args, **kwargs):
        if os.environ.get("ENABLE_FALL_BACK", "True") == "False":
            func(*args, **kwargs)

    return impl


def sot_only_test(func):
    """
    run this test function in ast only mode.
    Usage:

    class TestA (unittest.TestCase):
        @ast_only_test
        def test_ast_only(self):
            pass
    """

    def impl(*args, **kwargs):
        if os.environ.get("ENABLE_FALL_BACK", "True") == "True":
            func(*args, **kwargs)

    return impl
