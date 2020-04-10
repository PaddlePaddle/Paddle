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
import gast
import inspect
import numpy
import six
import textwrap
import threading
import warnings
from collections import defaultdict

from paddle.fluid import framework
from paddle.fluid import core, executor
from paddle.fluid.dygraph import guard, to_variable
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import convert_to_static
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import data_layer_not_check
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.data_feeder import check_type

__all__ = ['ProgramTranslator', 'convert_function_with_cache']


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        self._dycode_to_static_func = dict()
        self._static_func_to_transformer = dict()

    def get_or_cache_func(self, func):
        code = self._get_dedent_code_string(func)
        static_func = self._dycode_to_static_func.get(code, None)

        if static_func is None:
            static_func, dygraph_to_static_transformer = convert_to_static(func)
            self._dycode_to_static_func[code] = static_func
            self._static_func_to_transformer[
                static_func] = dygraph_to_static_transformer

        return static_func

    def get_transformer(self, func):
        return self._static_func_to_transformer.get(func, None)

    def _get_dedent_code_string(self, func):
        raw_code = inspect.getsource(func)
        dedent_code = textwrap.dedent(raw_code)
        return dedent_code

    def exist(self, func):
        return self._dycode_to_static_func.get(
            self._get_dedent_code_string(func), None) is not None


_CACHE_LOCK = threading.Lock()
_FUNCTION_CACHE = FunctionCache()


def convert_function_with_cache(dygraph_func):
    """
    Transform function of dygraph into static function using the cache mechanism.
    """
    with _CACHE_LOCK:
        static_func = _FUNCTION_CACHE.get_or_cache_func(dygraph_func)
        return static_func


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
        self._main_program = framework.default_main_program()
        self._startup_program = framework.default_startup_program()
        self._func_cache = FunctionCache()
        # Stores the entry function of Net or Model.
        self._forward_func = None
        self._feed_name_to_idx = {}
        self._is_repeated = False
        # Indicates whether the function call is still building program.
        # Because user can call recursively when `Net` has sub class in
        # `forward()`.
        self._in_build_process = True

    def build_program_and_return_output(self, dyfunc, *args, **kwargs):
        """
        Executes the main_program with specialized inputs so that the program
        is built. This method also return outputs of program as fetch_list
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
        static_func = self._func_cache.get_or_cache_func(dyfunc)
        # self._forward_func is entry function of Net or Model.
        # It can be called for multiple times, but layers from these functions
        # call stack will be added into self._main_program only once.
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
        with framework.program_guard(self._main_program, self._startup_program):
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
        with framework.program_guard(self._main_program, self._startup_program):
            for feed_name, idx in self.feed_name_to_idx.items():
                batch_data = args[idx]
                assert isinstance(
                    batch_data, numpy.ndarray
                ), "Input {} should be numpy.ndarray, but received {}.".format(
                    feed_name, type(batch_data))
                feed_layer = data_layer_not_check(
                    name=feed_name,
                    shape=list(batch_data.shape),
                    dtype=str(batch_data.dtype))
                self._inputs.append(feed_layer)

    def _get_name_to_idx(self, func):
        """
        Returns name and index of input args from `forward(args)`
        that need to be replaced with `fluid.data`.
        """
        transformer = self._func_cache.get_transformer(func)
        feed_name_to_idx = transformer.get_feed_name_to_idx()
        return feed_name_to_idx

    @property
    def main_program(self):
        return self._main_program

    @property
    def startup_program(self):
        return self._startup_program

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


class ProgramTranslator(object):
    _singleton_lock = threading.Lock()
    _instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance._initialized = False
            cls._instance.__init__()

    def __init__(self, exe=None, place=None):
        # To make sure that calls __init__ only once.
        if self._initialized:
            return
        self._initialized = True
        self._place = core.CPUPlace() if place is None else place
        if exe is None:
            self._exe = executor.Executor(self._place)
        else:
            self._exe = exe
        self._program_cache = ProgramCache()
        self._optimizer_info = None
        self._optimizer = None
        self._loss_name = None
        # Once main_program is changed, should run startup_program.
        self._need_startup = True

    def get_output(self, dygraph_func, *args, **kwargs):
        """
        Return the output tensors for dygraph function and its arguments
        """
        if in_dygraph_mode():
            warnings.warn(
                "The ProgramTranslator.get_output doesn't work in dygraph "
                "mode. We will just return dygraph output. Use it in "
                "static mode if you would like to translate to static graph.")
            return dygraph_func(*args, **kwargs)

        program_cache = self.get_program_cache()
        outputs = program_cache.build_program_and_return_output(dygraph_func,
                                                                *args, **kwargs)
        if not program_cache.in_build_process:
            outputs = self.run(*args, **kwargs)
            with guard():
                outputs = [to_variable(x) for x in outputs]
        return outputs

    def get_func(self, dygraph_func):
        """
        Return the translated static function from dygraph function
        """
        if in_dygraph_mode():
            warnings.warn(
                "The ProgramTranslator.get_func doesn't work in dygraph "
                "mode. We will just return dygraph function. Use it in "
                "static mode if you would like to translate to static graph.")
            return dygraph_func
        static_func = convert_function_with_cache(dygraph_func)
        return static_func

    def get_program(self, dygraph_func, *args, **kwargs):
        """
        Return the translated static program and input/output variables from
        dygraph function.
        """
        if in_dygraph_mode():
            warnings.warn(
                "The ProgramTranslator.get_program doesn't work in dygraph "
                "mode. We will just return dygraph output. Use it in static "
                "mode if you would like to translate to static graph.")
            return dygraph_func(*args, **kwargs)
        program_cache = self.get_program_cache()
        outputs = program_cache.build_program_and_return_output(dygraph_func,
                                                                *args, **kwargs)
        return self.main_program, self.startup_program, program_cache.inputs, outputs

    def get_code(self, dygraph_func):
        """
        Return the translated static function code from dygraph code
        """
        # Get AST from dygraph function
        raw_code = inspect.getsource(dygraph_func)
        code = textwrap.dedent(raw_code)
        root = gast.parse(code)

        # Transform AST
        dygraph_to_static = DygraphToStaticAst()
        root_wrapper = dygraph_to_static.get_static_ast(root)

        # Get source_code
        source_code = ast_to_source_code(root_wrapper.node)
        return source_code

    def run(self, *args, **kwargs):
        """
        Execute main_program and returns output Tensors.
        """
        feed_dict, fetch_list = self._prepare(args)

        main_program = self._program_cache.main_program
        outputs = self._exe.run(main_program,
                                feed=feed_dict,
                                fetch_list=fetch_list)

        return outputs

    def set_optimizer(self, optimizer, index_of_loss=0):
        """
        Support to set or update the optimizer used to minimize loss.
        """
        check_type(index_of_loss, "index_of_loss", int,
                   "ProgramTranslator.set_optimizer")
        self._check_cache_valid()
        if self._optimizer and self._loss_name:
            raise ValueError(
                "{} for {} has already been set before. Please confirm not to call `set_optimizer` in for loop. ".
                format(self._optimizer, self._loss_name))
        self._optimizer_info = (optimizer, index_of_loss)

    def save_inference_model(self, dirname, feed=None, fetch=None):
        """
        Save current model as the inference model.
        """
        program_cache = self.get_program_cache()
        if feed is None:
            feeded_var_names = [i.name for i in program_cache.inputs]
        else:
            feeded_var_names = [program_cache.inputs[i].name for i in feed]

        target_vars = program_cache.outputs
        from paddle.fluid.io import save_inference_model
        save_inference_model(
            dirname=dirname,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            executor=self._exe,
            main_program=self.main_program.clone())

    def _prepare(self, args):
        """
        Prepare with feed_dict, fetch_list, optimizer and initialize vars
        by running startup_program.
        """

        # Update batch_data for feed_dict
        feed_dict = self._update_batch_data(args)
        fetch_list = self._program_cache.outputs

        # Add optimizer if needed.
        if self._optimizer_info and self._optimizer is None:
            self._add_optimizer()

        if self._need_startup:
            self._exe.run(self.startup_program)
            self._need_startup = False

        return feed_dict, fetch_list

    def _check_cache_valid(self):
        """
        Check whether the current program is consistent with `default_main_program`.
        In some models and unittest, program will be switched frequently by `program_guard`.
        If does, the cached program and other properties are not available and should be reset.
        """
        if self._program_cache.main_program:
            if self._program_cache.main_program != framework.default_main_program(
            ):
                ProgramTranslator.reset()

    def _update_batch_data(self, args):
        """
        Update cached batch data while training program.
        """
        feed_name_to_idx = self._program_cache.feed_name_to_idx
        feed_vars = self._program_cache.inputs
        feed_dict = {}
        for feed_var in feed_vars:
            idx = feed_name_to_idx[feed_var.name]
            feed_dict[feed_var.name] = args[idx]

        return feed_dict

    def _add_optimizer(self):
        """
        Support to set or update the optimizer used to minimize loss.
        """
        optimizer, index_of_loss = self._optimizer_info

        outputs = self._program_cache.outputs
        outputs = [outputs] if not isinstance(outputs,
                                              (list, tuple)) else outputs

        assert abs(index_of_loss) < len(outputs), \
            "index_of_loss: {} shall not exceed the length of outputs: {}.".format(
            index_of_loss, len(outputs))

        loss_var = outputs[index_of_loss]
        check_type(loss_var, "loss_var", framework.Variable,
                   "ProgramTranslator._add_optimizer")

        main_program = self._program_cache.main_program
        startup_program = self._program_cache.startup_program
        all_vars = main_program.block(0).vars

        if all_vars.get(loss_var.name, None) is None:
            raise ValueError(
                "Can't find {} in main_program, please confirm whether the input loss is correct."
                .format(loss_var.name))
        # Add optimizer to minimize loss
        with framework.program_guard(main_program, startup_program):
            optimizer.minimize(loss_var)

        self._optimizer = optimizer
        self._loss_name = loss_var.name

    def get_program_cache(self):
        """
        Return the ProgramCache instance.
        """
        self._check_cache_valid()
        return self._program_cache

    @property
    def main_program(self):
        return self._program_cache.main_program

    @property
    def startup_program(self):
        return self._program_cache.startup_program
