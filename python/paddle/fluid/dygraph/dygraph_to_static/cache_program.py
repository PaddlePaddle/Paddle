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
import inspect
import textwrap
import threading
import numpy
import six

from paddle.fluid import framework
from paddle.fluid import core, executor
from paddle.fluid.data import data
from paddle.fluid.dygraph.dygraph_to_static import convert_to_static

__all__ = ['AutoTracer']


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        self._cache_funcs = dict()
        self._func_to_transformer = dict()

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

    def __init__(self):
        self._inputs = []
        self._outputs = []
        # Always set program to default_main_program. Because once `__call__` is called,
        # it means layers(or Ops) are added into default_main_program switched by outer
        # `with` statement.
        self._program = framework.default_main_program()
        self._func_cache = FunctionCache()
        # Stores the entry function of Net or Model.
        self._forward_func = None
        self._feed_name_to_idx = {}
        self._is_repeated = False
        # Indicates whether the function call is still building program.
        # Because `__call__` can be called recursively when `Net` has
        # sub class in `forward()`.
        self._in_build_process = True

    def __call__(self, dyfunc, *args, **kwargs):
        """
        Executes the main_program with specialized inputs.
        """
        # Transfroms dygraph function into static functions and caches them.
        static_func = self._transform_or_cache_layers(dyfunc)

        # 1. Adds `fluid.data` layers for input if needed
        if not self._inputs:
            self._add_feed_layers(args, kwargs)

        # 2. Avoids inserting forward ops repeatedly.
        if self._is_repeated:
            return self.outputs

        # 3. Builds program only once and returns the output Variables.
        outputs = self._get_or_build_program(static_func, args, kwargs)

        if static_func == self._forward_func:
            self._in_build_process = False

        return outputs

    def _transform_or_cache_layers(self, dyfunc):
        """
        Transforms dygraph function into static function.
        """
        static_func = self._func_cache(dyfunc)
        # self._forward_func is entry function of Net or Model.
        # It can be called for multiple times, but layers from these functions
        # call stack will be added into self._program only once.
        # After that, cached program will be always returned by default.
        if static_func == self._forward_func:
            self._is_repeated = True

        if self._forward_func is None:
            self._forward_func = static_func

        return static_func

    def _get_or_build_program(self, func, args, kwargs):
        """
        Returns program of the input function. If called at first time,
        builds a new program and caches it.
        """
        with framework.program_guard(self._program):
            if func == self._forward_func:
                # Replaces input data with `layers.data`
                args = list(args)
                for feed_layer in self._inputs:
                    idx = self.feed_name_to_idx[feed_layer.name]
                    args[idx] = feed_layer
                fetch_list = func(*args, **kwargs)
                self._outputs = fetch_list
            else:
                fetch_list = func(*args, **kwargs)

        return fetch_list

    def _add_feed_layers(self, args, kwargs):
        """
        Adds `fluid.data` if the input `numpy.ndarray` is converted into `Variable`
        by `to_variable()`, it makes program to be executed dynamically.
        """
        if not self._feed_name_to_idx:
            self._feed_name_to_idx = self._get_name_to_idx(self._forward_func)
        with framework.program_guard(self._program):
            for feed_name, idx in self.feed_name_to_idx.items():
                batch_data = args[idx]
                assert isinstance(
                    batch_data, numpy.ndarray
                ), "Input {} should be numpy.ndarray, but received {}.".format(
                    feed_name, type(batch_data))
                feed_layer = data(
                    name=feed_name,
                    shape=[-1] + list(batch_data.shape[1:]),
                    dtype=str(batch_data.dtype))
                self._inputs.append(feed_layer)

    def _get_name_to_idx(self, func):
        """
        Returns name and index of input args from `forward(args)`
        that need to be replaced with `fluid.data`.
        """
        transformer = self._func_cache.transformer(func)
        feed_name_to_idx = transformer.get_feed_name_to_idx()
        return feed_name_to_idx

    @property
    def program(self):
        return self._program

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def feed_name_to_idx(self):
        return self._feed_name_to_idx

    @property
    def in_build_process(self):
        return self._in_build_process


class AutoTracer(object):

    _instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance.__initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("FuncProgram hasn\'t been created!")
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance.__initialized = False
            cls._instance.__init__()

    def __init__(self, exe=None, place=None):
        # To make sure that calls __init__ only once.
        if self.__initialized:
            return
        self.__initialized = True
        self._place = core.CPUPlace() if place is None else place
        if exe is None:
            self._exe = executor.Executor(self._place)
        else:
            self._exe = exe
        self._cached_program = ProgramCache()
        self._optimizer = None
        self._already_minimized = False
        # Once main_program is changed, should run startup_program.
        self._need_startup = True

    def run(self, *args, **kwargs):
        """
        Executes main_program and returns output Tensors.
        """
        feed_dict, fetch_list = self._prepare(args)

        main_program = self._cached_program.program
        outputs = self._exe.run(main_program,
                                feed=feed_dict,
                                fetch_list=fetch_list)

        return outputs

    def _prepare(self, args):
        """
        Prepares with feed_dict, fetch_list, optimizer and initialize vars
        by running startup_program.
        """

        # Updates batch_data for feed_dict
        feed_dict = self._update_batch_data(args)
        fetch_list = self._cached_program.outputs

        # Adds optimizer if needed.
        if self._optimizer and not self._already_minimized:
            self._add_optimizer()

        if self._need_startup:
            self._exe.run(framework.default_startup_program())
            self._need_startup = False

        return feed_dict, fetch_list

    def _check_cache_valid(self):
        """
        Checks whether the current program is consistent with `default_main_program`.
        In some models and unittest, program will be switched frequently by `program_guard`.
        If does, the cached program and other properties are not available and should be reset.
        """
        if self._cached_program.program:
            if self._cached_program.program != framework.default_main_program():
                AutoTracer.reset()

    def _update_batch_data(self, args):
        """
        Updates cached batch data while training program.
        """
        feed_name_to_idx = self._cached_program.feed_name_to_idx
        feed_vars = self._cached_program.inputs
        feed_dict = {}
        for feed_var in feed_vars:
            idx = feed_name_to_idx[feed_var.name]
            feed_dict[feed_var.name] = args[idx]

        return feed_dict

    def set_optimizer(self, optimizer, loss_name):
        """
        Supports to set or update the optimizer used to minimize loss.
        """
        self._check_cache_valid()
        self._optimizer = optimizer

        if not isinstance(loss_name, six.string_types):
            raise ValueError(
                "Type of input loss_name should type(str), but received {}.".
                format(type(loss_name)))
        self._loss_name = loss_name

    def _add_optimizer(self):
        """
        Supports to set or update the optimizer used to minimize loss.
        """
        main_program = self._cached_program.program
        all_vars = main_program.block(0).vars
        loss_var = all_vars.get(self._loss_name, None)

        if loss_var is None:
            raise ValueError(
                "Can't find {} in main_program, please confirm whether the loss input is correct"
                .format(self._loss_name))
        # Adds optimizer to minimize loss
        with framework.program_guard(main_program):
            self._optimizer.minimize(loss_var)

        # Avoids to set optimizer repeatedly.
        self._already_minimized = True

    def get_cached_program(self):
        """
        Returns the ProgramCache instance.
        """
        self._check_cache_valid()
        return self._cached_program

    @property
    def program(self):
        return self._cached_program.program
