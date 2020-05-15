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
import inspect
import logging
import textwrap
import threading
import collections
import numpy as np
from paddle.fluid import core, scope_guard
from paddle.fluid import framework
from paddle.fluid import executor
from paddle.fluid import unique_name
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import convert_to_static
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.base import param_guard
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph.dygraph_to_static.partial_program import partial_program_from

__all__ = ['ProgramTranslator', 'convert_function_with_cache']

logger = logging.getLogger("fluid")


class FunctionCache(object):
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        self._dycode_to_static_func = dict()
        self._static_func_to_transformer = dict()

    def get_or_cache_func(self, func):
        # code = self._get_dedent_code_string(func)
        static_func = self._dycode_to_static_func.get(func, None)

        if static_func is None:
            static_func, dygraph_to_static_transformer = convert_to_static(func)
            self._dycode_to_static_func[func] = static_func
            self._static_func_to_transformer[
                func] = dygraph_to_static_transformer

        return static_func

    def get_transformer(self, func):
        return self._static_func_to_transformer.get(func, None)

    def _get_dedent_code_string(self, func):
        raw_code = inspect.getsource(func)
        dedent_code = textwrap.dedent(raw_code)
        return dedent_code

    def exist(self, func):
        return self._dycode_to_static_func.get(func, None) is not None


_CACHE_LOCK = threading.Lock()
_FUNCTION_CACHE = FunctionCache()


def convert_function_with_cache(dygraph_func):
    """
    Transforms function of dygraph into static function using the cache mechanism.
    """
    with _CACHE_LOCK:
        static_func = _FUNCTION_CACHE.get_or_cache_func(dygraph_func)
        return static_func


class FunctionSpec(object):
    def __init__(self, func, args, kwargs):
        self._dyfunc = func
        self._args = args
        self._kwargs = kwargs

    def is_method(self):
        return self._args and isinstance(self._args[0], layers.Layer)

    def parameters(self, include_sublayer=True):
        params = collections.OrderedDict()
        if self.is_method():
            if include_sublayer:
                params = self._args[0].parameters()
                names = [p.name for p in params]
                params = collections.OrderedDict(zip(names, params))
            else:
                params = self._args[0]._parameters
        return params

    @switch_to_static_graph
    def to_static_inputs(self, main_program):
        inputs = []
        block = main_program.global_block()
        for input_var in self.args:
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
        return inputs

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
            return self._dyfunc, self._args[0]
        else:
            return self._dyfunc

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == self.__key()


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
        dygaph_function = func_spec.dyfunc
        static_func = convert_function_with_cache(dygaph_function)

        main_program, startup_program = framework.Program(), framework.Program()
        # Note: The random seed should be synchronized into cached program
        # if set in `fluid.dygrap_guard` because some ops rely on it, such as
        # `fluid.layers.dropout`.
        main_program.random_seed = framework.default_main_program().random_seed
        startup_program.random_seed = framework.default_startup_program(
        ).random_seed

        with framework.program_guard(main_program, startup_program):
            # 1. Adds `fluid.data` layers for input if needed
            inputs = func_spec.to_static_inputs(main_program)

            # 2. Gets all ParamBases in the function
            all_parameters = list(func_spec.parameters().values())

            # 3. Builds program only once and returns the output Variables.
            with param_guard(func_spec.parameters(False)):
                outputs = static_func(*inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs] if outputs else []

        return ConcreteProgram(
            inputs=inputs,
            outputs=outputs,
            parameters=all_parameters,
            func=dygaph_function,
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
            logger.info(
                "The ProgramTranslator.get_output doesn't work when setting ProgramTranslator.enable = False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)

        function_spec = FunctionSpec(dygraph_func, args, kwargs)
        _, partial_program_layer = self._program_cache[function_spec]

        if args and isinstance(args[0], layers.Layer):
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
            logger.info(
                "The ProgramTranslator.get_func doesn't work when setting ProgramTranslator.enable=False. We will "
                "just return dygraph output.")
            return dygraph_func

        static_func = convert_function_with_cache(dygraph_func)
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
            logger.info(
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
        raw_code = inspect.getsource(dygraph_func)
        code = textwrap.dedent(raw_code)
        root = gast.parse(code)

        # Transform AST
        dygraph_to_static = DygraphToStaticAst()
        root_wrapper = dygraph_to_static.get_static_ast(root)

        # Get source_code
        source_code = ast_to_source_code(root_wrapper.node)
        return source_code

    def save_inference_model(self, dirname, feed=None, fetch=None):
        """
        Saves current model as the inference model. It will prune the main_program
        to build a new program especially for inference, and then save it and all
        related parameters to given `dirname` . The saved inference model can be
        loaded by `:ref:`api_fluid_io_load_inference_model` or `C++ inference APIs.

        Args:
            dirname (str): the directory to save the inference model.
            feed (list[int], optional): the indices of the input variables of the
                dygraph functions which will be saved as input variables in
                inference model. If None, all input variables of the dygraph function
                would be the inputs of the saved inference model. Default None.
            fetch (list[int], optional): the indices of the returned variable of the
                dygraph functions which will be saved as output variables in
                inference model. If None, all output variables of the dygraph function
                would be the outputs of the saved inference model. Default None.
        Returns:
            None
        Examples:
            .. code-block:: python
                import numpy as np
                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear
                from paddle.fluid.dygraph import declarative
                from paddle.fluid.dygraph import ProgramTranslator

                class SimpleNet(fluid.dygraph.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = Linear(in_size, out_size)

                    @declarative
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        loss = fluid.layers.mean(z)
                        return z, loss

                with fluid.dygraph.guard(fluid.CPUPlace()):
                    net = SimpleNet(8, 8)
                    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
                    x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                    for i in range(10):
                        loss, out = net(x)
                        loss.backward()
                        adam.minimize(loss)
                        net.clear_gradients()
                # Save inference model.
                # Note that fetch=[0] means we set 'z' as the inference output.
                prog_trans = ProgramTranslator()
                prog_trans.save_inference_model("./dy2stat_infer_model", fetch=[0])

                # In this example, the inference model will be pruned based on output (z).
                # The pruned inference program is going to be saved in the folder
                # "./dy2stat_infer_model" and parameters are going to be saved in separate
                # files in the folder.
        """

        def get_feed_fetch(var_list, partial_vars, return_name=False):
            vars = [
                var for var in var_list if isinstance(var, framework.Variable)
            ]
            if partial_vars:
                vars = [vars[idx] for idx in partial_vars]
            if return_name:
                vars = [var.name for var in vars]

            return vars

        func_spec, (concrete_program,
                    partial_layer) = self._program_cache.last()
        # share paramBase data with parameter
        scope = core.Scope()
        for param_base in concrete_program.parameters:
            param_tensor = scope.var(param_base.name).get_tensor()
            src_tensor = param_base.value().get_tensor()
            param_tensor._share_data_with(src_tensor)

        feed_var_names = get_feed_fetch(concrete_program.inputs, feed, True)
        fetch_vars = get_feed_fetch(concrete_program.outputs, fetch)

        from paddle.fluid.io import save_inference_model
        with scope_guard(scope):
            save_inference_model(
                dirname=dirname,
                feeded_var_names=feed_var_names,
                target_vars=fetch_vars,
                executor=executor.Executor(framework._current_expected_place()),
                main_program=concrete_program.main_program.clone())

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
