# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

import gast
import numpy as np
from paddle.fluid import core
from paddle.fluid import executor
from paddle.fluid import framework
from paddle.fluid import scope_guard
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph.base import param_guard
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
from paddle.fluid.dygraph.dygraph_to_static.error import ERROR_DATA
from paddle.fluid.dygraph.dygraph_to_static.error import attach_error_data
from paddle.fluid.dygraph.dygraph_to_static.origin_info import attach_origin_info
from paddle.fluid.dygraph.dygraph_to_static.origin_info import create_and_update_origin_info_map
from paddle.fluid.dygraph.dygraph_to_static.origin_info import update_op_callstack_with_origin_info
from paddle.fluid.dygraph.dygraph_to_static.partial_program import partial_program_from
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_func
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import func_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import unwrap
from paddle.fluid.layers.utils import flatten
from paddle.fluid.layers.utils import pack_sequence_as
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

__all__ = ['ProgramTranslator', 'convert_to_static']


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        # Caches the converted static functions. {dygraph_func: static_func}
        self._converted_static_func_caches = dict()
        # Caches the converted ast node for same source code. {source_code: ast_root}
        self._code_to_ast_caches = dict()
        self._dygraph_to_static = DygraphToStaticAst()

    def convert_with_cache(self, func):
        """
        Returns the cached static function or converts it when first encounters the function.
        """
        # If hit cache, return it directly.
        static_func = self._converted_static_func_caches.get(func, None)

        if static_func is None:
            static_func = self._convert(func)
            self._converted_static_func_caches[func] = static_func

        return static_func

    def _convert(self, func):
        """
        Converts dygraph function into static function. For two functions with same dedent code,
        the second function will reuse the transformed ast node of previous one.

        For example:
            # A.py
            def foo(x, y):
                z = x + y
                return z

            # B.py
            def foo(x, y):
                z = x + y
                return z

        If the conversion of A.foo happens after B.foo, it will reuse the transformed ast node of B.foo
        to speed up the conversion.
        """
        # Note: In Python2, it will raise OSError when inspect function
        # with decorator directly and function.__wrapped__ holds the actual function.
        func = unwrap(func)
        source_code = func_to_source_code(func)

        # TODO(liym27):
        #  Consider this case: source_code in self._code_to_ast_caches,
        #  but actually they are methods in different classes.
        #  Maybe use (__class__, source_code) as key
        if source_code in self._code_to_ast_caches:
            root_wrapper = self._code_to_ast_caches[source_code]
        else:
            root = gast.parse(source_code)
            root = attach_origin_info(root, func)
            root_wrapper = self._dygraph_to_static.get_static_ast(root)
            self._code_to_ast_caches[source_code] = root_wrapper

        # Get static function from AST
        static_func, file_name = ast_to_func(root_wrapper.node, func)

        create_and_update_origin_info_map(root_wrapper.node, static_func)
        return static_func

    def exist(self, func):
        return func in self._converted_static_func_caches


_CACHE_LOCK = threading.Lock()
_FUNCTION_CACHE = FunctionCache()


def convert_to_static(function):
    """
    Transforms function of dygraph into static function using the cache mechanism.

    Args:
        function(callable): The function with dygraph layers that will be converted into static layers.
    """
    with _CACHE_LOCK:
        static_func = _FUNCTION_CACHE.convert_with_cache(function)
        return static_func


class FunctionSpec(object):
    def __init__(self, func, args, kwargs):
        self._dyfunc = func
        self._args = args
        self._kwargs = kwargs

        # TODO(liym27): func has multi layer decorator
        dyfunc = getattr(func, '__wrapped__', func)
        self._dyfunc_code = inspect.getsource(dyfunc)

    def is_method(self):
        return self._args and isinstance(self._args[0], layers.Layer)

    def parameters(self, include_sublayer=True):
        """
        Returns parameters of decorated layers. If set `include_sublayer` True,
        the parameters created in sub layers will be added.
        """
        params = collections.OrderedDict()
        if self.is_method():
            layer_instance = self._args[0]
            if include_sublayer:
                params = layer_instance.parameters()
                names = [p.name for p in params]
                params = collections.OrderedDict(zip(names, params))
            else:
                params = layer_instance._parameters
        return params

    def buffers(self, include_sublayer=True):
        """
        Returns Variable buffers of decorated layers. If set `include_sublayer` True,
        the Variable buffers created in sub layers will be added.
        """
        buffers = collections.OrderedDict()
        if self.is_method():
            layer_instance = self._args[0]
            if include_sublayer:
                buffers = layer_instance.buffers()
                names = [buffer.name for buffer in buffers]
                buffers = collections.OrderedDict(zip(names, buffers))
            else:
                buffers = layer_instance._buffers
        return buffers

    @switch_to_static_graph
    def to_static_inputs(self, main_program):
        inputs = []
        block = main_program.global_block()
        for input_var in flatten(self.args):
            if isinstance(input_var, np.ndarray):
                feed_layer = block.create_var(
                    name=unique_name.generate('feed'),
                    shape=list(input_var.shape),
                    dtype=input_var.dtype,
                    is_data=True,
                    need_check_feed=False)
            elif isinstance(input_var, core.VarBase):
                feed_layer = block.create_var(
                    name=input_var.name,
                    shape=list(input_var.shape),
                    dtype=input_var.dtype,
                    stop_gradient=input_var.stop_gradient,
                    need_check_feed=False)
            else:
                feed_layer = input_var

            inputs.append(feed_layer)
        # Restores the nested structure as self.args
        return pack_sequence_as(self.args, inputs)

    @property
    def dyfunc(self):
        return self._dyfunc

    @property
    def args(self):
        return self._args

    def __key(self):
        # Note: if dygraph function is a method of class,
        # consider instance info as hash key.
        if self.is_method():
            # NOTE: we can use Layer's (instance + function code) as hash key.
            # An instance will not hold two identical methods 
            return self._dyfunc_code, self._args[0]
        else:
            return self._dyfunc

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == self.__key()


# Flag that indicates whether running code under `@declarative`
_in_declarative_mode_ = False


def in_declarative_mode():
    """
    Return a bool value that indicates whether running code under `@declarative`

    """
    return _in_declarative_mode_


@signature_safe_contextmanager
def _switch_declarative_mode_guard_(is_declarative=True):

    global _in_declarative_mode_
    original_val = _in_declarative_mode_
    _in_declarative_mode_ = is_declarative
    yield
    _in_declarative_mode_ = original_val


class ConcreteProgram(object):
    def __init__(self,
                 inputs,
                 outputs,
                 parameters,
                 func,
                 main_program,
                 startup_program=None):
        self.inputs = inputs
        self.outputs = outputs
        self.main_program = main_program
        self.startup_program = startup_program
        self.parameters = parameters
        self.func_spec = func

    @staticmethod
    @switch_to_static_graph
    def from_func_spec(func_spec):
        """
        Builds the main_program with specialized inputs and returns outputs
        of program as fetch_list.
        """
        # Transforms dygraph function into static function and caches it.
        dygraph_function = func_spec.dyfunc
        static_func = convert_to_static(dygraph_function)

        main_program, startup_program = framework.Program(), framework.Program()
        # Note: The random seed should be synchronized into cached program
        # if set in `fluid.dygraph_guard` because some ops rely on it, such as
        # `fluid.layers.dropout`.
        main_program.random_seed = framework.default_main_program().random_seed
        startup_program.random_seed = framework.default_startup_program(
        ).random_seed

        with framework.program_guard(main_program, startup_program):
            with _switch_declarative_mode_guard_(is_declarative=True):
                # 1. Adds `fluid.data` layers for input if needed
                inputs = func_spec.to_static_inputs(main_program)

                # 2. Gets all ParamBases and buffered VarBases in the function
                all_parameters_and_buffers = list(func_spec.parameters().values(
                )) + list(func_spec.buffers().values())

                # 3. Builds program only once and returns the output Variables.
                with param_guard(func_spec.parameters(False)), param_guard(
                        func_spec.buffers(False)):
                    try:
                        outputs = static_func(*inputs)
                    except BaseException as e:
                        # NOTE: If e is raised in compile time, e should be attached to ERROR_DATA here.
                        attach_error_data(e)
                        raise

                if not isinstance(outputs,
                                  (tuple, list)) and outputs is not None:
                    outputs = [outputs]

        main_program = update_op_callstack_with_origin_info(main_program)

        return ConcreteProgram(
            inputs=inputs,
            outputs=outputs,
            parameters=all_parameters_and_buffers,
            func=dygraph_function,
            main_program=main_program,
            startup_program=startup_program)


class ProgramCache(object):
    """
    Wrapper class for the program functions defined by dygraph function.
    """

    def __init__(self):
        self._caches = collections.OrderedDict()

    def _build_once(self, func_spec):
        concrete_program = ConcreteProgram.from_func_spec(func_spec)
        return concrete_program, partial_program_from(concrete_program)

    def __getitem__(self, item):
        if not isinstance(item, FunctionSpec):
            raise ValueError(
                'type(item) should be FunctionSpec, but received %s' %
                type(item))
        if item not in self._caches:
            self._caches[item] = self._build_once(item)
        return self._caches[item]

    def get_program(self, item):
        if not isinstance(item, FunctionSpec):
            raise ValueError(
                "Input item's type should be FunctionSpec, but received %s" %
                type(item))
        if item not in self._caches:
            raise RuntimeError(
                "Failed to find program for input item, please decorate input function by `@declarative`."
            )
        return self._caches[item]

    def last(self):
        assert len(
            self._caches) >= 1, "No valid cached program in ProgramCache."
        key = next(reversed(self._caches.keys()))
        return key, self._caches[key]


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class ProgramTranslator(object):
    """
    Class to translate dygraph function into static graph function. The object
    of this class is a singleton.

    Args:
        None.

    Returns:
        ProgramTranslator: the singleton object.

    Examples:
        .. code-block:: python

        import paddle.fluid as fluid

        # Two methods get same object because ProgramTranslator is a singleton
        fluid.dygraph.ProgramTranslator()
        fluid.dygraph.ProgramTranslator.get_instance()

    """

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

    def __init__(self):
        # To make sure that calls __init__ only once.
        if self._initialized:
            return
        self._initialized = True
        self._program_cache = ProgramCache()
        self.enable_declarative = True

    def enable(self, enable_declarative):
        """
        Enable or disable the converting from imperative to declarative by
        ProgramTranslator globally.

        Args:
            enable_declarative (bool): True or False to enable or disable declarative.

        Returns:
            None.

        Examples:
            .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            @fluid.dygraph.jit.declarative
            def func(x):
                x = fluid.dygraph.to_variable(x)
                if fluid.layers.mean(x) > 0:
                    x_v = x - 1
                else:
                    x_v = x + 1
                return x_v

            prog_trans = fluid.dygraph.ProgramTranslator()
            prog_trans.enable(False)

            x = np.ones([1, 2])
            # The declarative is disabled so the func is run in dygraph
            with fluid.dygraph.guard():
                print(func(x).numpy()) # [[2. 2.]]

        """
        check_type(enable_declarative, "enable_declarative", bool,
                   "ProgramTranslator.enable")
        self.enable_declarative = enable_declarative

    def get_output(self, dygraph_func, *args, **kwargs):
        """
        Returns the output dygraph VarBase for dygraph function. The dygraph
        function will be translated into static graph function so the under
        beneath numerical result will be calculated by declarative mode.

        Args:
            dygraph_func (callable): the dygraph function.
            *args, **kwargs : the input argument of dygraph_func.

        Returns:
            VarBase or tuple of VarBase: the dygraph VarBase containing digital
                result.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                def func(x):
                    x = fluid.dygraph.to_variable(x)
                    if fluid.layers.mean(x) > 0:
                        x_v = x - 1
                    else:
                        x_v = x + 1
                    return x_v

                prog_trans = fluid.dygraph.ProgramTranslator()

                with fluid.dygraph.guard():
                    x = np.ones([1, 2])
                    x_v = prog_trans.get_output(func, x)
                    print(x_v.numpy()) # [[0. 0.]]

        """
        assert callable(
            dygraph_func
        ), "Input dygraph_func is not a callable in ProgramTranslator.get_output"
        if not self.enable_declarative:
            warnings.warn(
                "The ProgramTranslator.get_output doesn't work when setting ProgramTranslator.enable = False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)

        function_spec = FunctionSpec(dygraph_func, args, kwargs)
        concrete_program, partial_program_layer = self._program_cache[
            function_spec]

        if args and isinstance(args[0], layers.Layer):
            # Synchronize self.training attribute.
            partial_program_layer.training = args[0].training
            args = args[1:]
        try:
            return partial_program_layer(args)

        except BaseException as e:
            # NOTE:
            # 1. If e is raised in compile time, e should have been attached to ERROR_DATA before;
            # 2. If e raised in runtime, e should be attached to ERROR_DATA here.
            if not hasattr(e, ERROR_DATA):
                # runtime error
                attach_error_data(e, in_runtime=True)
            raise

    def get_func(self, dygraph_func):
        """
        Returns a callable function which converts imperative dygraph APIs of
        the input dygraph_func into declarative net-building APIs, which means
        it doesn't return immediate digital result as get_output does.
        Users should handle Program and Executor by themselves.

        Args:
            dygraph_func (callable): the dygraph function.

        Returns:
            callable: converting imperative dygraph APIs into declarative
            net-building APIs.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                def func(x):
                    x = fluid.dygraph.to_variable(x)
                    if fluid.layers.mean(x) > 0:
                        x_v = x - 1
                    else:
                        x_v = x + 1
                    return x_v

                prog_trans = fluid.dygraph.ProgramTranslator()

                static_func = prog_trans.get_func(func)
                print(callable(static_func)) # True

        """
        assert callable(
            dygraph_func
        ), "Input dygraph_func is not a callable in ProgramTranslator.get_func"
        if not self.enable_declarative:
            warnings.warn(
                "The ProgramTranslator.get_func doesn't work when setting ProgramTranslator.enable=False. We will "
                "just return dygraph output.")
            return dygraph_func

        static_func = convert_to_static(dygraph_func)
        return static_func

    def get_program(self, dygraph_func, *args, **kwargs):
        """
        Returns the translated static program and input/output variables from
        dygraph function. The users can use the program to run by executor.

        Args:
            dygraph_func (callable): the dygraph function.
            *args, **kwargs : the input argument of dygraph_func.

        Returns:
            tuple of (main_program, startup_program, inputs, outputs) whose
            types are (Program, Program, list of Variable, list of Variable).
            main_program: the converted main program.
            startup_program: the converted startup program.
            inputs: list of input Variables which need to be fed.
            outputs: list of output Variables which users can fetch.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                def func(x):
                    x = fluid.dygraph.to_variable(x)
                    if fluid.layers.mean(x) > 0:
                        x_v = x - 1
                    else:
                        x_v = x + 1
                    return x_v

                prog_trans = fluid.dygraph.ProgramTranslator()

                x = np.ones([1, 2])
                main_prog, start_prog, inputs, outputs = prog_trans.get_program(func, x)
                print([i.name for i in inputs])
                # ['feed_0'] the feed input variable name representing x
                print([o.name for o in outputs])
                # ['_generated_var_4'] the fetch output variable name representing x_v        

        """
        assert callable(
            dygraph_func
        ), "Input dygraph_func is not a callable in ProgramTranslator.get_program"
        if not self.enable_declarative:
            warnings.warn(
                "The ProgramTranslator.get_program doesn't work when setting ProgramTranslator.enable=False."
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)

        func_spec = FunctionSpec(dygraph_func, args, kwargs)
        concrete_program, _ = self._program_cache[func_spec]
        # Note: concrete_program hold all input/output infos include non-Variable
        input_vars = [
            var for var in concrete_program.inputs
            if isinstance(var, framework.Variable)
        ]
        output_vars = [
            var for var in concrete_program.outputs
            if isinstance(var, framework.Variable)
        ]

        return concrete_program.main_program, \
               concrete_program.startup_program, \
               input_vars, \
               output_vars

    def get_code(self, dygraph_func):
        """
        Returns the translated static function string code from dygraph function.

        Args:
            dygraph_func (callable): the dygraph function.

        Returns:
            str: the string code of translated static function.

        Examples:
            .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            def func(x):
                x = fluid.dygraph.to_variable(x)
                if fluid.layers.mean(x) > 0:
                    x_v = x - 1
                else:
                    x_v = x + 1
                return x_v

            prog_trans = fluid.dygraph.ProgramTranslator()

            code = prog_trans.get_code(func)
            print(type(code)) # <class 'str'>

        """
        assert callable(
            dygraph_func
        ), "Input dygraph_func is not a callable in ProgramTranslator.get_code"
        # Gets AST from dygraph function

        unwrap_func = unwrap(dygraph_func)
        raw_code = inspect.getsource(unwrap_func)
        code = textwrap.dedent(raw_code)
        root = gast.parse(code)

        # Transform AST
        dygraph_to_static = DygraphToStaticAst()
        root_wrapper = dygraph_to_static.get_static_ast(root)

        # Get source_code
        source_code = ast_to_source_code(root_wrapper.node)
        return source_code

    def get_program_cache(self):
        """
        Returns the ProgramCache instance. This method is used by PaddlePaddle
        developers to manage program cache in ProgramTranslator. Normal users
        don't have to call this method.

        Returns:
            ProgramCache: ProgramCache instance of ProgramTranslator.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog_trans = fluid.dygraph.ProgramTranslator()
                prog_cache = prog_trans.get_program_cache()

        """
        return self._program_cache
