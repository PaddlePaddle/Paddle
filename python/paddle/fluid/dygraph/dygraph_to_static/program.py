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
import gast
import six
import numpy
from dygraph_to_static.ast_transformer import DygraphToStaticAst
from dygraph_to_static.ast_utils import ast_to_func

import paddle.fluid as fluid


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions to the same function.
    """

    def __init__(self):
        self._cache_funcs = collections.OrderedDict()

    def __call__(self, func):
        static_func = self._get_or_transform_func(func)
        return static_func

    def _get_or_transform_func(self, func):

        cache_key = self.hash_key(func)
        static_func = self._cache_funcs.get(cache_key, None)

        if static_func is None:
            print("create new func:")
            print(func.__name__)
            static_func = self._convert_to_static(func)
            self._cache_funcs[cache_key] = static_func

        return static_func

    def _convert_to_static(self, func):
        raw_code = inspect.getsource(func)
        code = textwrap.dedent(raw_code)
        root = gast.parse(code)
        root_wrapper = DygraphToStaticAst().get_static_ast(root)

        func_name = func.__name__
        static_func, file_name = ast_to_func(root_wrapper.node, func_name)
        return static_func

    def hash_key(self, func):
        raw_code = inspect.getsource(func)
        code = textwrap.dedent(raw_code)

        return hash(code)

    def exist(self, func):
        return self._cache_funcs.get(self.hash_key(func), None) is not None


class FuncProgram(object):
    """
    Wrapper class for the program functions defined by dygraph function.
    """

    def __init__(self, name, program=None, need_startup=True):
        self.inputs = []
        self.outputs = []
        self._name = name
        self._func_cache = FunctionCache()
        # TODO: Currently, default_main_program is used to initialize _program.
        # It's a trick way because only one program is executed while the training
        # process starts in most time. However, more than one program will be built
        # such as GAN. The following code should be optimized.
        if program is not None and isinstance(program, fluid.Program):
            self._program = program
        else:
            self._program = fluid.default_main_program()
        self._first_call = True
        # Stores the function entry of Net or Model.
        self._forward_func = None
        # Traces the function call stack
        self._traced_funcs = []
        self.func_outputs_cache = {}
        self._is_repeated = False
        self.batch_data = []
        self._need_startup = True

    def __call__(self, *args, **kwargs):
        """
        Calls the main_program specialized to the inputs.
        """

        if fluid.in_dygraph_mode():
            return self._traced_funcs[-1](*args, **kwargs)

        self._prepare(args, kwargs)

        last_func = self._traced_funcs.pop()

        if self._is_repeated:
            if self._is_build_process(args, kwargs):
                return self.func_outputs_cache[last_func]
            return self._run(args, kwargs)

        outputs = self._get_or_build_program(last_func, args, kwargs)
        self.func_outputs_cache[last_func] = outputs

        if last_func == self._forward_func and not self._is_build_process(
                args, kwargs):
            outputs = self._run(args, kwargs)

        return outputs

    def _is_build_process(self, args, kwargs):
        if args is not None:
            for arg in args:
                if isinstance(arg, numpy.ndarray):
                    return False
        return True

    def _prepare(self, args, kwargs):
        """
        Prepares batch_data and Adds `fluid.data` layer into program.
        """
        if not self.inputs:
            self._add_placeholder(args, kwargs)

        if self._traced_funcs[
                -1] == self._forward_func and not self._is_build_process(
                    args, kwargs):
            self.batch_data = [args[1], args[2]]

    def get_program(self):
        return self._program

    def add_layers(self, dyfunc):
        """
        Transforms dygraph function into static function, and adds layers into
        main_program.
        """
        is_cached = self._func_cache.exist(dyfunc)
        static_func = self._func_cache(dyfunc)
        # self._forward_func is function entry of Net or Model.
        # It can be called multiple times, but layers from thess function
        # call stack will be added into self._program only once.
        # After that, cached program will be always returned by default.
        if static_func == self._forward_func:
            self._is_repeated = True

        if self._forward_func is None:
            self._forward_func = static_func

        self._traced_funcs.append(static_func)

        return static_func

    def _get_or_build_program(self, func, args, kwargs):
        """
        Returns program for the inputs. If called at first time, 
        builds a new program and caches it.
        """
        with fluid.program_guard(self._program):
            if func == self._forward_func and not self._is_build_process(
                    args, kwargs):
                args = list(args)
                args[1:] = self.inputs[:]
                fetch_list = func(*args, **kwargs)
                adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
                adam.minimize(fetch_list[-1])
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
        exe = fluid.Executor(fluid.CPUPlace())
        if self._need_startup:
            self._need_startup = False
            exe.run(fluid.default_startup_program())
        feed = dict(zip(input_names, self.batch_data))
        res = exe.run(self._program, feed=feed, fetch_list=self.outputs)
        return res

    def _parse_feed_fetch(self, args, kwargs):
        """
        Returns feed and fetch_list.
        """
        feed = {}
        fetch_list = []

        return feed, fetch_list

    def _add_placeholder(self, args, kwargs):
        """
        Adds `fluid.data` if input args numpy.ndarray, it make program
        to be executed dynamically.
        """
        assert len(self._traced_funcs) > 0, "No traced function is available."
        func = self._traced_funcs[-1]
        fullargspec = inspect.getfullargspec(func)

        with fluid.program_guard(self._program):
            if args is not None:
                for index, arg in enumerate(args):
                    arg_name = fullargspec.args[index]
                    # TODO: input maybe don't need stop_gradient.
                    if isinstance(arg, numpy.ndarray):
                        placeholder = fluid.data(
                            name=arg_name,
                            shape=[None] + list(arg.shape[1:]),
                            dtype=arg.dtype)
                        self.inputs.append(placeholder)
