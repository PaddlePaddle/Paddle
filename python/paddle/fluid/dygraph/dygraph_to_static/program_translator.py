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
import gast
import logging
import inspect
import warnings
import textwrap
import threading
import collections
from paddle.fluid import scope_guard
from paddle.fluid import framework
from paddle.fluid import executor
from paddle.fluid import unique_name
from paddle.fluid.dygraph import layers
from paddle.fluid.layers.utils import flatten
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import func_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import type_name
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_func
from paddle.fluid.dygraph.dygraph_to_static.input_spec import TensorSpec, FunctionSpec
from paddle.fluid.dygraph.dygraph_to_static.input_spec import get_buffers, get_parameters
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle.fluid.dygraph.base import param_guard
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph.dygraph_to_static.partial_program import partial_program_from
from paddle.fluid.annotations import deprecated

__all__ = ['ProgramTranslator', 'convert_to_static']

MAX_TRACED_PROGRAM_COUNT = 5


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
        func = getattr(func, '__wrapped__', func)
        source_code = func_to_source_code(func)
        if source_code in self._code_to_ast_caches:
            root_wrapper = self._code_to_ast_caches[source_code]
        else:
            root = gast.parse(source_code)
            root_wrapper = self._dygraph_to_static.get_static_ast(root)
            self._code_to_ast_caches[source_code] = root_wrapper

        # Get static function from AST
        static_func, file_name = ast_to_func(root_wrapper.node, func)
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


class CacheKey(object):
    def __init__(self, function_spec, inputs_with_spec, class_instance):
        self.function_spec = function_spec
        # TODO: make sure it can hash 
        self.inputs_with_spec = inputs_with_spec
        self.class_instance = class_instance

    @classmethod
    def from_func_and_args(cls, function_spec, args, kwargs, class_instance):
        if args and isinstance(args[0], layers.Layer):
            args = args[1:]
        # 2. convert tensor and numpy array into TensorSpec 
        _args, _kwargs = function_spec.unified_args_and_kwargs(args, kwargs)
        inputs_with_spec = function_spec.args_to_variable_spec(_args, _kwargs)

        # 3. check whether hit the cache or build a new program for the input arguments
        return CacheKey(function_spec, inputs_with_spec, class_instance)

    def __hash__(self):
        # TODO(Aurelius84): list, dict is not hashable. Currently just consider tensorSpec as hash key.
        tensor_specs = [
            input for input in flatten(self.inputs_with_spec)
            if isinstance(input, TensorSpec)
        ]

        return hash(
            (id(self.function_spec), tuple(tensor_specs), self.class_instance))

    def __eq__(self, other):
        return (type(self) is type(other)) and hash(self) == hash(other)

    def __neq__(self, other):
        return not self == other

    def __repr__(self):
        return "id(function_spec): {}, inputs_with_spec: {}, class_instance: {}".format(
            id(self.function_spec), self.inputs_with_spec, self.class_instance)


def unwrap(func):
    """
    Unwraps a decorated function and returns the decorator list and inner target.
    """
    decorators = []
    cur = func
    while True:
        if isinstance(cur, PartialProgram):
            decorators.append(cur)
            cur = cur.dygraph_function
        else:  # TODO: deal with other decorators
            break
    return decorators, cur


class PartialProgram(object):
    def __init__(self, dygraph_func, input_spec=None):
        self._dygraph_func = dygraph_func
        self._input_spec = input_spec
        self._function_spec = FunctionSpec(dygraph_func, input_spec)
        self._program_cache = ProgramCache()
        # save the instance `self` while decorating a method of class.
        self._class_instance = None

    def __get__(self, instance, owner):
        self._class_instance = instance
        return self

    def __call__(self, *args, **kwargs):
        # 1. call dygraph function if not enable `declarative`
        if not program_trans.enable_declarative:
            return self._call_dygraph_func(*args, **kwargs)

        # 2. convert tensor and numpy array into TensorSpec 
        args, kwargs = self._function_spec.unified_args_and_kwargs(args, kwargs)
        inputs_with_spec = self._function_spec.args_to_variable_spec(args,
                                                                     kwargs)

        # TODO: whether we need this class or just a functionSpec
        # 3. check whether hit the cache or build a new program for the input arguments
        cache_key = CacheKey(self._function_spec, inputs_with_spec,
                             self._class_instance)
        # print(inputs_with_spec, hash(cache_key))
        _, partial_program_layer = self._program_cache[cache_key]

        # 4. # Synchronize self.training attribute.
        if isinstance(self._class_instance, layers.Layer):
            partial_program_layer.training = self._class_instance.training

        return partial_program_layer(args)

    def _call_dygraph_func(self, *args, **kwargs):
        """
        Calls dygraph function directly and returns the outputs.
        """
        if self._class_instance is not None:
            dygraph_func = self._dygraph_func.__get__(self._class_instance)
        else:
            dygraph_func = self._dygraph_func

        return dygraph_func(*args, **kwargs)

    def get_trace_count(self):
        return len(self._program_cache)

    @property
    def to_code(self):
        # TODO
        return None

    @property
    def dygraph_function(self):
        return self._dygraph_func

    @property
    def concrete_program(self):
        # if specific the `input_spec`, the length of program_cache will always 1,
        # else, return the last one.
        if len(self._program_cache) > 0:
            cache_key, (concrete_program,
                        partial_layer) = self._program_cache.last()
            return concrete_program
        else:
            raise ValueError("No valid transformed program for function: {}".
                             fromat(self._dygraph_func.__name__))

    @property
    def program_cache(self):
        return self._program_cache


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
    def from_func_spec(func_spec, inputs_spec, class_instance):
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
                # inputs = func_spec.to_static_inputs(main_program)
                # TODO: modify it into plain function if don't rely on function.spec
                inputs = func_spec.to_static_inputs_with_signature(inputs_spec,
                                                                   main_program)
                if class_instance:
                    inputs = tuple([class_instance] + list(inputs))

                # 2. Gets all ParamBases and buffered VarBases in the function
                all_parameters_and_buffers = list(
                    get_parameters(class_instance).values()) + list(
                        get_buffers(class_instance).values())

                # 3. Builds program only once and returns the output Variables.
                with param_guard(get_parameters(
                        class_instance, False)), param_guard(
                            get_buffers(class_instance, False)):
                    print(inputs)
                    outputs = static_func(*inputs)
                if not isinstance(outputs,
                                  (tuple, list)) and outputs is not None:
                    outputs = [outputs]

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

    def _build_once(self, cache_key):
        print("build program for : ", cache_key)
        concrete_program = ConcreteProgram.from_func_spec(
            func_spec=cache_key.function_spec,
            inputs_spec=cache_key.inputs_with_spec,
            class_instance=cache_key.class_instance)
        return concrete_program, partial_program_from(concrete_program)

    def __getitem__(self, item):
        if not isinstance(item, CacheKey):
            raise ValueError('type(item) should be CacheKey, but received %s' %
                             type_name(item))

        print("hash {} in self._caches: {}".format(
            hash(item), item in self._caches))

        if item not in self._caches:
            self._caches[item] = self._build_once(item)
            # raise warnings if number of traced program is out of `max_tracing_count`
            current_tracing_count = len(self._caches)
            if current_tracing_count > MAX_TRACED_PROGRAM_COUNT:
                logging.warning(
                    "Current traced program number: {} > `max_tracing_count`:{}. Too much cached programs will bring expensive overhead. The reason may be: (1) passing tensors with different shapes, (2) passing python objects instead of tensors.".
                    format(current_tracing_count, MAX_TRACED_PROGRAM_COUNT))

        return self._caches[item]

    def get_program(self, item):
        if not isinstance(item, CacheKey):
            raise ValueError(
                "Input item's type should be FunctionSpec, but received %s" %
                type_name(item))
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

    def __len__(self):
        return len(self._caches)

    def concrete_programs(self):
        return [cp for key, (cp, _) in self._caches.iteritems()]


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

        function_spec = FunctionSpec(dygraph_func)
        cache_key = CacheKey.from_func_and_args(function_spec, args, kwargs,
                                                getattr(dygraph_func,
                                                        '__self__', None))
        _, partial_program_layer = self._program_cache[cache_key]

        if args and isinstance(args[0], layers.Layer):
            # Synchronize self.training attribute.
            partial_program_layer.training = args[0].training
            args = args[1:]

        return partial_program_layer(args)

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

        function_spec = FunctionSpec(dygraph_func)
        cache_key = CacheKey.from_func_and_args(function_spec, args, kwargs,
                                                getattr(dygraph_func,
                                                        '__self__', None))
        concrete_program, partial_program_layer = self._program_cache[cache_key]

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
        raw_code = inspect.getsource(dygraph_func)
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


program_trans = ProgramTranslator()
