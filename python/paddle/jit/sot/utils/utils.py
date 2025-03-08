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

import builtins
import inspect
import sys
import time
import types
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from weakref import WeakValueDictionary

import numpy as np

import paddle
from paddle.utils import flatten, map_structure

from .envs import (
    ENV_SOT_LOG_LEVEL,
    ENV_STRICT_MODE,
)
from .paddle_api_config import (
    break_graph_set,
    paddle_api_list,
    paddle_api_module_prefix,
)

if TYPE_CHECKING:
    from paddle._typing import NestedStructure

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
ConstTypes = (int, float, str, bool, type(None))


class Singleton(type):
    _instances: dict[Any, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class NameGenerator:
    def __init__(self, prefix):
        self.counter = 0
        self.prefix = prefix

    def next(self):
        name = self.prefix + str(self.counter)
        self.counter += 1
        return name

    def match_name(self, name: str) -> bool:
        return name.startswith(self.prefix)


class SymbolRegistry:
    def __init__(self):
        self.symbol_generator = NameGenerator(prefix="___t_")
        self.tmp_names_record = OrderedDict()
        self.declared_symbols: set[str] = set()
        self.symbol_table = {}

    def next_symbol(self) -> str:
        return self.symbol_generator.next()

    def request_symbol(self, expr: str) -> str:
        if expr in self.symbol_table:
            return self.symbol_table[expr]
        symbol = self.next_symbol()
        self.symbol_table[expr] = symbol
        return symbol

    def gen_expr(self, expr: str, gen_expr_fn):
        symbol = self.symbol_table[expr]
        if symbol in self.declared_symbols:
            return symbol
        self.declared_symbols.add(symbol)
        return f"({symbol} := ({gen_expr_fn()}))"


_symbol_registry = SymbolRegistry()


@contextmanager
def switch_symbol_registry():
    global _symbol_registry
    original_registry = _symbol_registry
    _symbol_registry = SymbolRegistry()
    yield
    _symbol_registry = original_registry


def current_symbol_registry():
    global _symbol_registry
    return _symbol_registry


class ResumeFnNameFactory(metaclass=Singleton):
    def __init__(self) -> None:
        self.gen = NameGenerator('resume_')

    def next(self):
        name = self.gen.next()
        return name


def log(level, *args):
    cur_level = ENV_SOT_LOG_LEVEL.get()
    if level <= cur_level:
        print(*args, end="", flush=True)


def log_do(level, fn):
    cur_level = ENV_SOT_LOG_LEVEL.get()
    if level <= cur_level:
        fn()


def log_format(level, str, *args):
    cur_level = ENV_SOT_LOG_LEVEL.get()
    if level <= cur_level:
        print(str.format(*args), end="", flush=True)


def log_enabled(level):
    return level <= ENV_SOT_LOG_LEVEL.get()


def no_eval_frame(func):
    def no_eval_frame_func(*args, **kwargs):
        old_cb = paddle.framework.core.set_eval_frame(None)
        try:
            retval = func(*args, **kwargs)
        except:
            raise
        finally:
            paddle.framework.core.set_eval_frame(old_cb)
        return retval

    return no_eval_frame_func


def is_comprehensive_name(name):
    return name in ["<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"]


def is_paddle_api(func):
    if isinstance(func, paddle.nn.Layer):  # ignore all the classes
        return False
    if hasattr(func, "__self__"):  # ignore all the methods
        return False
    if inspect.isclass(
        func
    ):  # paddle.Tensor should not be wrapped, but how about other situations?
        return False
    return in_paddle_module(func) or func in paddle_api_list


def is_builtin_fn(fn):
    special_builtin_fns = [weakref.ref]
    if fn in special_builtin_fns:
        return True
    if isinstance(fn, types.BuiltinFunctionType):
        return True
    for member_name, member in inspect.getmembers(builtins):
        if member is fn and isinstance(member, type):
            return True
    return False


def in_paddle_module(func):
    if hasattr(func, "__module__"):
        module_str = func.__module__
        if module_str is None:
            return False
        log(5, "find paddle function with __module__: ", module_str, "\n")
        if hasattr(func, "__name__"):
            log(
                5, "                     with __name__  : ", func.__name__, "\n"
            )
        log(5, "                     with results   : ")
        for prefix in paddle_api_module_prefix:
            if module_str.startswith(prefix):
                log(5, " True\n")
                return True
    log(5, " False\n")
    return False


def is_break_graph_api(func):
    return func in break_graph_set


def map_if(
    *structures: NestedStructure[T1],
    pred: Callable[[T1], bool],
    true_fn: Callable[[T1], T2],
    false_fn: Callable[[T1], T3],
) -> NestedStructure[T2 | T3]:
    def replace(*args):
        if pred(*args):
            return true_fn(*args)
        return false_fn(*args)

    return map_structure(replace, *structures)


def flatten_extend(structure):
    for item in flatten(structure):
        if isinstance(item, slice):
            yield item.start
            yield item.stop
            yield item.step
        else:
            yield item


def map_if_extend(structure, pred, true_fn, false_fn):
    """support extended structures like slice and SliceVariable"""

    def wrapped_pred(x):
        if isinstance(x, slice):
            return True
        return pred(x)

    def wrapped_true_fn(x):
        if isinstance(x, (slice)):
            l = [x.start, x.stop, x.step]
            l = map_if_extend(l, pred, true_fn, false_fn)
            return slice(*l)
        return true_fn(x)

    return map_if(
        structure, pred=wrapped_pred, true_fn=wrapped_true_fn, false_fn=false_fn
    )


def count_if(*structures, pred):
    def is_true(*args):
        if pred(*args):
            return 1
        return 0

    return sum(flatten(map_structure(is_true, *structures)))


class Cache:
    def __init__(self, weak=False):
        if not weak:
            self.cache = {}
        else:
            self.cache = WeakValueDictionary()
        self.hit_num = 0

    def __call__(self, *args, **kwargs):
        cache_key = self.key_fn(*args, **kwargs)
        if not hashable(cache_key):
            return self.value_fn(*args, **kwargs)
        if cache_key in self.cache:
            log(5, "cache hit: ", cache_key, "\n")
            self.hit_num += 1
            return self.cache[cache_key]
        value = self.value_fn(*args, **kwargs)
        self.cache[cache_key] = value
        return value

    def clear(self):
        self.cache.clear()
        self.hit_num = 0

    def key_fn(self, *args, **kwargs):
        raise NotImplementedError

    def value_fn(self, *args, **kwargs):
        raise NotImplementedError


def execute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execute time:", execution_time)
        return result

    return wrapper


def meta_str(shape, dtype, stop_gradient):
    return f"(shape: {shape}, dtype: {dtype}, stop_gradient: {stop_gradient})"


def is_strict_mode():
    return ENV_STRICT_MODE.get()


def list_find_index_by_id(li: list[Any], item: Any) -> int:
    return [id(it) for it in li].index(id(item))


def list_contain_by_id(li: list[Any], item: Any) -> int:
    return id(item) in [id(it) for it in li]


def get_unbound_method(obj, name):
    # TODO(dev): Consider the case of patching methods to instances
    return getattr(obj.__class__, name)


class SotUndefinedVar(metaclass=Singleton):
    pass


def hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError as e:
        return False


def printable(obj):
    try:
        str(obj)
        return True
    except Exception as e:
        return False


class StepInfo:
    BACK_TRACE_STEPS = 20

    def __init__(self):
        self.step_count = -1

    def need_back_trace(self):
        return self.step_count < self.BACK_TRACE_STEPS


class StepInfoManager(metaclass=Singleton):
    def __init__(self):
        self.step_record = {}
        self.current_code = None
        self.current_step_info = None

    @contextmanager
    def step_guard(self, code):
        try:
            old_code = self.current_code
            old_info = self.current_step_info

            self.current_code = code
            if code not in self.step_record:
                self.step_record[code] = StepInfo()
            self.current_step_info = self.step_record[code]

            self.current_step_info.step_count += 1
            yield
        finally:
            self.current_code = old_code
            self.current_step_info = old_info

    @property
    def need_back_trace(self):
        return self.current_step_info.need_back_trace()

    @property
    def current_step(self):
        return self.current_step_info.step_count

    def clear(self):
        self.step_record.clear()
        self.current_code = None
        self.current_step = -1


def get_api_fullname(api):
    api_name = api.__name__
    module_str = api.__module__
    while len(module_str) > 0:
        module = sys.modules[module_str]
        if hasattr(module, api_name):
            return module_str + "." + api_name
        module_str = module_str.rpartition(".")[0]
    return None


def get_numpy_ufuncs():
    ufuncs = [
        ufunc
        for _, ufunc in inspect.getmembers(
            np, lambda member: isinstance(member, np.ufunc)
        )
    ]
    unary_ufuncs = filter(lambda ufunc: ufunc.nin == 1, ufuncs)
    binary_ufuncs = filter(lambda ufunc: ufunc.nin == 2, ufuncs)
    return list(unary_ufuncs), list(binary_ufuncs)
