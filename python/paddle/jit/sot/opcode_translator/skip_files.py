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

import _collections_abc
import _weakrefset
import abc
import codecs
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import sys
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import uuid
import warnings
import weakref

import decorator
import google.protobuf
import numpy
import setuptools

import paddle

NEED_SKIP_THIRD_PARTY_MODULES = {
    abc,
    collections,
    contextlib,
    copy,
    copyreg,
    dataclasses,
    enum,
    functools,
    google.protobuf,
    importlib,
    inspect,
    linecache,
    logging,
    multiprocessing,
    numpy,
    operator,
    os,
    posixpath,
    random,
    re,
    selectors,
    signal,
    tempfile,
    threading,
    tokenize,
    traceback,
    types,
    typing,
    unittest,
    weakref,
    _collections_abc,
    _weakrefset,
    decorator,
    codecs,
    uuid,
    setuptools,
    warnings,
}

if sys.version_info < (3, 11):
    import sre_compile
    import sre_parse

    NEED_SKIP_THIRD_PARTY_MODULES.add(sre_compile)
    NEED_SKIP_THIRD_PARTY_MODULES.add(sre_parse)

if sys.version_info < (3, 12):
    import distutils

    NEED_SKIP_THIRD_PARTY_MODULES.add(distutils)


def _strip_init_py(s):
    return re.sub(r"__init__.py$", "", s)


def _module_dir(m: types.ModuleType):
    return _strip_init_py(m.__file__)


skip_file_names = {_module_dir(m) for m in NEED_SKIP_THIRD_PARTY_MODULES}


sot_path = os.path.dirname(__file__).rpartition(os.sep)[0] + os.sep
paddle_path = sys.modules["paddle"].__file__.rpartition(os.sep)[0] + os.sep

skip_file_names.add(sot_path)
skip_file_names.add(paddle_path)
skip_file_names.add(
    "<frozen importlib",
)
skip_file_names.add("<__array_function__ internals>")

skip_file_name_re = re.compile(
    f"^({'|'.join(map(re.escape, skip_file_names))})"
)

no_skip_code = {paddle.nn.Sequential.forward.__code__}

with_graph_codes = (
    paddle.nn.Layer.__call__.__code__,
    paddle.nn.Layer._dygraph_call_func.__code__,
)


def setup_skip_files():
    paddle.framework.core.eval_frame_skip_file_prefix(tuple(skip_file_names))
    paddle.framework.core.eval_frame_no_skip_codes(tuple(no_skip_code))
    paddle.framework.core.sot_setup_codes_with_graph(with_graph_codes)
