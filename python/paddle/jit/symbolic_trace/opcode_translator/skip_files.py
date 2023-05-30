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
import sre_compile
import sre_parse
import sys
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import uuid
import weakref

import _collections_abc
import _weakrefset
import decorator
import numpy


def _strip_init_py(s):
    return re.sub(r"__init__.py$", "", s)


def _module_dir(m: types.ModuleType):
    return _strip_init_py(m.__file__)


skip_file_names = {
    _module_dir(m)
    for m in (
        abc,
        collections,
        contextlib,
        copy,
        copyreg,
        dataclasses,
        enum,
        functools,
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
        sre_compile,
        sre_parse,
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
    )
}


symbolic_trace_path = os.path.dirname(__file__).rpartition("/")[0] + "/"
paddle_path = sys.modules["paddle"].__file__.rpartition("/")[0] + "/"

skip_file_names.add(symbolic_trace_path)
skip_file_names.add(paddle_path)
skip_file_names.add(
    "<frozen importlib",
)
skip_file_names.add("<__array_function__ internals>")

skip_file_name_re = re.compile(
    f"^({'|'.join(map(re.escape, skip_file_names))})"
)


def need_skip_path(filepath):
    if not filepath.startswith("<"):
        filepath = os.path.abspath(filepath)
    return bool(skip_file_name_re.match(filepath))
