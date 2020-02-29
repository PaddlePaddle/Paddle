#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import collections
import inspect
import textwrap
import threading
import numpy

from paddle.fluid import framework
from paddle.fluid.layers import io
from paddle.fluid import core, executor
from paddle.fluid.dygraph.dygraph_to_static import convert_to_static

__all__ = ['ProgramCache']


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions to the same function.
    """

    def __init__(self):
        self._cache_funcs = collections.OrderedDict()

        self._func_to_transformer = collections.OrderedDict()

    def __call__(self, func):
        static_func = self._get_or_cache_func(func)
        return static_func

    def _get_or_cache_func(self, func):

        cache_key = self.hash_key(func)
        static_func = self._cache_funcs.get(cache_key, None)

        if static_func is None:
            static_func, dygraph_to_static = convert_to_static(func)
            self._cache_funcs[cache_key] = static_func
            self._func_to_transformer[static_func] = dygraph_to_static

        return static_func

    def transformer(self, func):
        return self._func_to_transformer.get(func, None)

    def hash_key(self, func):
        raw_code = inspect.getsource(func)
        code = textwrap.dedent(raw_code)

        return hash(code)

    def exist(self, func):
        return self._cache_funcs.get(self.hash_key(func), None) is not None


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class ProgramCache(object):
    """
    Wrapper class for the program functions defined by dygraph function.
    """

    _instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            raise ValueError("FuncProgram hasn\'t been created!")
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance.__init__()

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.batch_data = []
        self.func_cache = FunctionCache()
        self.program = None
        # Stores the function entry of Net or Model.
        self.forward_func = None
        # Traces the function call stack
        self.traced_funcs = []
        self.func_outputs_cache = {}
        self.is_repeated = False
        self.need_startup = True
        self.already_minimize = False
        self.optimizer = None
        self.is_build_process = True

    def __call__(self, *args, **kwargs):
        """
        Calls the main_program specialized to the inputs.
        """

        # 1. Adds `fluid.data` layers for input or updates batch_data
        self._prepare(args, kwargs)

        # 2. Run cached program to avoid inserting forward ops repeatedly.
        if self.is_repeated:
            return self._run(args, kwargs)

        # 3. Build program once.
        last_func = self.traced_funcs.pop()
        outputs = self._get_or_build_program(last_func, args, kwargs)
        self.func_outputs_cache[last_func] = outputs

        if last_func == self.forward_func and not self.is_build_process:
            outputs = self._run(args, kwargs)

        return outputs

    def _check_cache_valid(self):
        """
        Checks whether the current program is consistent with `default_main_program`.
        In some models and unittest, program will be switched frequently by `program_guard`.
        If does, the cached information is not available and should be reset.
        """
        if self.program:
            if self.program != framework.default_main_program():
                ProgramCache.reset()
        # Always set program to default_main_program. Because once `__call__` is called,
        # it means layers(or Ops) are added into default_main_program switched by outer
        # `with` statement.
        self.program = framework.default_main_program()

    def _prepare(self, args, kwargs):
        """
        Prepares batch_data and Adds `fluid.data` layer into program.
        """
        if not self.inputs:
            self._add_feed_layers(args, kwargs)
        elif self.traced_funcs[-1] == self.forward_func:
            self._update_batch_data(args, kwargs)

    def add_layers(self, dyfunc):
        """
        Transforms dygraph function into static function, and adds layers into
        main_program.
        """
        self._check_cache_valid()
        static_func = self.func_cache(dyfunc)
        # self._forward_func is entry function of Net or Model.
        # It can be called multiple times, but layers from thess function
        # call stack will be added into self._program only once.
        # After that, cached program will be always returned by default.
        if static_func == self.forward_func:
            self.is_repeated = True

        if self.forward_func is None:
            self.forward_func = static_func

        self.traced_funcs.append(static_func)

        return static_func

    def _get_or_build_program(self, func, args, kwargs):
        """
        Returns program for the inputs. If called at first time,
        builds a new program and caches it.
        """
        with framework.program_guard(self.program):
            if func == self.forward_func and self.is_build_process:
                # Replaces input data with `layers.data`
                feed_name_to_idx = self._feed_name_to_idx(self.forward_func)
                args = list(args)
                for feed_layer in self.inputs:
                    idx = feed_name_to_idx[feed_layer.name]
                    args[idx] = feed_layer
                self.is_build_process = False
                fetch_list = func(*args, **kwargs)
                self._add_optimizer(fetch_list)
            else:
                fetch_list = func(*args, **kwargs)
        self.outputs = fetch_list
        return fetch_list

    def _run(self, args, kwargs):
        """
        Executes the main_program and returns Tensor of fetch_list
        """
        assert self.inputs, "inputs is not initialized."
        assert self.batch_data, "batch_data is empty."

        input_names = [in_var.name for in_var in self.inputs]
        exe = executor.Executor(core.CPUPlace())
        if self.need_startup:
            self.need_startup = False
            exe.run(framework.default_startup_program())
        feed_dict = dict(zip(input_names, self.batch_data))
        res = exe.run(self.program, feed=feed_dict, fetch_list=self.outputs)
        return res

    def set_optimizer(self, optimizer, force_update=True):
        """
        Supports to set or update the optimizer used to minimize loss.
        """
        self._check_cache_valid()

        if not self.optimizer or force_update:
            self.optimizer = optimizer
        else:
            raise ValueError("optimizer already has been set.")

    def _add_optimizer(self, fetch_list):
        """
        Adds optimizer to minimize loss.
        """
        if not isinstance(fetch_list, (list, tuple)):
            fetch_list = [fetch_list]
        if not self.already_minimize and self.optimizer:
            for out_var in fetch_list:
                # TODO: How to determine which vars is referred to loss.
                if numpy.product(out_var.shape) == 1 and 'mean' in out_var.name:
                    self.optimizer.minimize(out_var)
                    self.already_minimize = True

    def _add_feed_layers(self, args, kwargs):
        """
        Adds `fluid.data` if the input `numpy.ndarray` is converted into `Variable`
        by `to_variable()`, it make program to be executed dynamically.
        """
        feed_name_to_idx = self._feed_name_to_idx(self.forward_func)
        with framework.program_guard(self.program):
            for feed_name, idx in feed_name_to_idx.items():
                batch_data = args[idx]
                assert isinstance(
                    batch_data, numpy.ndarray
                ), "Input {} should be numpy.ndarray, but received {}.".format(
                    feed_name, type(batch_data))
                feed_layer = io.data(
                    name=feed_name,
                    shape=list(batch_data.shape[1:]),
                    dtype=str(batch_data.dtype))
                self.inputs.append(feed_layer)
                self.batch_data.append(batch_data)

    def _update_batch_data(self, args, kwargs):
        """
        Updates cached batch data while training program.
        """
        feed_name_to_idx = self._feed_name_to_idx(self.forward_func)
        prev_len = len(self.batch_data)
        self.batch_data = []
        for feed_layer in self.inputs:
            feed_name = feed_layer.name
            idx = feed_name_to_idx[feed_name]
            self.batch_data.append(args[idx])

        assert len(self.batch_data) == prev_len

    def _feed_name_to_idx(self, func):
        """
        Returns names and index of input args from `forward(args)`
        that need to be replaced with `fluid.data`.
        """
        transformer = self.func_cache.transformer(func)
        feed_name_to_idx = transformer.get_feed_name_to_idx()
        return feed_name_to_idx
