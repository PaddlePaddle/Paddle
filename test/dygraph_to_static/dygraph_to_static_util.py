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


def run_both(cls):
    """
    run dy2static by ast and fallback
    """
    for key in dir(cls):
        if key.startswith("test"):
            test_func = getattr(cls, key)

            setattr(cls, key + "_ast", to_ast(test_func))

    return cls


def run_ast(cls):
    """
    run dy2staic by ast
    """

    for key in dir(cls):
        if key.startswith("test"):
            test_func = getattr(cls, key)
            setattr(cls, key, to_ast(test_func))

    return cls
