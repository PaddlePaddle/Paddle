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

import functools
import weakref
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from weakref import WeakValueDictionary

import paddle
from paddle.jit.dy2static.utils import parameters_persistent_mode_is_enabled
from paddle.jit.utils import OrderedSet
from paddle.utils import flatten, map_structure

from ..utils import (
    InnerError,
    NameGenerator,
    Singleton,
    flatten_extend,
    get_api_fullname,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


_StatementContextT = TypeVar("_StatementContextT", bound="StatementContext")


class ParametersHolder:
    def __init__(self):
        self._params = WeakValueDictionary[
            str, paddle.base.framework.EagerParamBase
        ]()

    def set(self, name, param):
        self._params[name] = param

    def get(self, name):
        if (param := self._params.get(name)) is None:
            raise InnerError(
                f"Parameter '{name}' not found in ParametersHolder."
            )
        return param

    def copy(self):
        new_holder = ParametersHolder()
        new_holder._params = self._params.copy()
        return new_holder


class Reference:  # to unify weak_ref and strong_ref
    def __init__(self, value, is_weak):
        self.is_weak = is_weak
        if is_weak is True:
            self.ref = weakref.ref(value)
        else:
            self.ref = value

    def __call__(self):
        if self.is_weak is True:
            return self.ref()
        else:
            return self.ref


class Symbol:
    """
    Symbol is used to distinguish a string and a `math variable`.
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __deepcopy__(self, memo=None):
        return Symbol(self.name)


class StatementContext: ...


class StatementContextRegistry:

    _ctx_map: dict[
        type[Any],
        Callable[[Any], AbstractContextManager[None]],
    ] = {}

    @classmethod
    def register_context_guard(
        cls,
        ctx_cls: type[_StatementContextT],
        handler: Callable[[_StatementContextT], AbstractContextManager[None]],
    ):
        """
        Register a context handler for the given context.
        """
        if ctx_cls in cls._ctx_map:
            raise ValueError(f"Context {ctx_cls} is already registered.")
        cls._ctx_map[ctx_cls] = handler

    @classmethod
    def register_context(
        cls,
        handler: Callable[[_StatementContextT], AbstractContextManager[None]],
    ):
        def decorator(ctx_cls: type[_StatementContextT]):
            cls.register_context_guard(ctx_cls, handler)
            return ctx_cls

        return decorator

    @classmethod
    def get_context_guard(
        cls,
        ctx_cls: type[_StatementContextT],
    ) -> Callable[[_StatementContextT], AbstractContextManager[None]]:
        """
        Get the context handler for the given context.
        """
        if ctx_cls not in cls._ctx_map:
            raise ValueError(f"Context {ctx_cls} is not registered.")
        return cls._ctx_map[ctx_cls]


class Statement:
    """
    Statement is used to represent a sentence of code for building the neural network model,
    which has four types: "call", "api", "method", and "layer".

    Note:
        Statement temporarily does not support control flow.
    """

    def __init__(
        self,
        type: str,
        name: str,
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        assert type in ["call", "api", "method", "layer", "AST"]
        self.name = name
        self.inputs = inputs  # (list of Symbols, dict of Symbols)
        self.outputs = outputs  # list of Symbol | PythonObj
        self.contexts = contexts  # list of StatementContext
        self.stmt_stack = (
            stacks  # a list of string to record the source code callstack.
        )
        self.type = type

    def __str__(self):
        return "{} || {} = {} ({}) ".format(
            self.type + " " * (10 - len(self.type)),
            self.to_string(self.outputs),
            self.name,
            self.to_string(self.inputs),
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def to_string(inps):
        return ", ".join(repr(x) for x in flatten(inps))


class CallStatement(Statement):
    def __init__(
        self,
        name: str,
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        super().__init__("call", name, inputs, outputs, contexts, stacks)
        self.sir_name = name


class ApiStatement(Statement):
    def __init__(
        self,
        api: Callable,
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        fullname = get_api_fullname(api)
        if fullname is None:
            fullname = "paddle." + api.__name__
        super().__init__("api", fullname, inputs, outputs, contexts, stacks)
        self.api = api


class MethodStatement(Statement):
    def __init__(
        self,
        name: str,
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        super().__init__("method", name, inputs, outputs, contexts, stacks)
        self.method = name


class LayerStatement(Statement):
    def __init__(
        self,
        layer: Reference,  # Reference of paddle.nn.Layer
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        if isinstance(layer, Reference):
            name = layer().__class__.__name__
        else:
            name = layer.__class__.__name__
        super().__init__(
            "layer",
            name,
            inputs,
            outputs,
            contexts,
            stacks,
        )
        self.layer = layer


class ASTStatement(Statement):
    def __init__(
        self,
        static_function,
        inputs: list[Symbol],
        outputs: list[Symbol],
        contexts: list[StatementContext],
        stacks: list[str],
    ):
        # this dygraph_function always has attr __code__, which is checked before
        dygraph_func = static_function.dygraph_function
        super().__init__(
            "AST",
            dygraph_func.__code__.co_name,
            inputs,
            outputs,
            contexts,
            stacks,
        )
        converted_func = paddle.jit.dy2static.convert_to_static(dygraph_func)
        func_self = getattr(dygraph_func, '__self__', None)
        if func_self is not None:
            converted_func = functools.partial(converted_func, func_self)
        self.converted_func = converted_func


class StatementIR:
    """
    StatementIR is the carrier that records the code for building the neural network model.It is
    a representation of a purely computational structure, and does not care about specific values.
    The function converted from StatementIR can ensure that it can be turned into a static state.
    In this way, we can reuse the original `to_static` function to realize the execution of the static graph.

    Note:
        Don't create by yourself, just use the StatementIRFactory.create()
    """

    def __init__(self, name: str):
        self.name = name
        self.inputs: list[Symbol] = []
        self.params: list[Symbol] = []
        self.outputs: list[Symbol] = []
        self.statements: list[Statement] = []

        self.symbol_meta_map = {}
        self.param_symbol = set()
        self.non_param_symbol = set()

    @property
    def input_with_params(self):
        return self.inputs + self.params

    def __len__(self):
        return len(self.statements)

    def __deepcopy__(self, memo=None):
        new_sir = StatementIR(self.name)
        new_sir.inputs = list(self.inputs)
        new_sir.params = list(self.params)
        new_sir.outputs = list(self.outputs)
        new_sir.statements = list(self.statements)
        new_sir.symbol_meta_map = dict(self.symbol_meta_map.items())
        new_sir.param_symbol = set(self.param_symbol)
        new_sir.non_param_symbol = set(self.non_param_symbol)
        return new_sir

    def set_parameter_info(self, params, non_params):
        self.param_symbol.update(params)
        self.non_param_symbol.update(non_params)

    def set_symbol_meta_map(self, meta_map):
        # if the meta of a input symbol inplace changed, we should get the origin meta as input of SIR
        meta_map.update(self.symbol_meta_map)
        self.symbol_meta_map = meta_map

    def add_input(self, input):
        self.inputs.append(input)

    def add_output(self, output):
        self.outputs.append(output)

    def add_statement(self, statement):
        assert isinstance(statement, Statement)
        self.statements.append(statement)

    def analyse_inputs(self):
        used_symbols = OrderedSet()
        generated_symbols = OrderedSet()
        for stmt in self.statements:
            for inp in flatten_extend(stmt.inputs):
                if isinstance(inp, Symbol) and inp not in generated_symbols:
                    used_symbols.add(inp)
            for out in flatten_extend(stmt.outputs):
                if isinstance(out, Symbol):
                    generated_symbols.add(out)

        used_symbols = sorted(used_symbols, key=lambda x: x.name)
        if not parameters_persistent_mode_is_enabled():
            return used_symbols, []
        input_symbols = [
            symbol for symbol in used_symbols if symbol not in self.param_symbol
        ]
        param_symbols = [
            symbol for symbol in used_symbols if symbol in self.param_symbol
        ]
        return input_symbols, param_symbols

    def __str__(self):
        strs = []
        strs.append(f"StatementIR: {self.name}")
        strs.append(f"  inputs: {map_structure(lambda x: x.name, self.inputs)}")
        strs.append(f"  params: {map_structure(lambda x: x.name, self.params)}")
        strs.append(
            f"  outputs: {map_structure(lambda x: x.name, self.outputs)}"
        )
        strs.append("  statements: ")
        for stmt in self.statements:
            strs.append(f"    {stmt}")
        return "\n".join(strs)

    def __repr__(self):
        return self.__str__()


class StatementIRFactory(metaclass=Singleton):
    """
    It is used to create a StatementIR.
    """

    def __init__(self):
        self.cache = {}
        self.name_generator = NameGenerator("SIR_")

    def __getitem__(self, key):
        return self.cache[key]

    def create(self, input_name=None):
        if input_name:
            name = input_name
        else:
            name = self.name_generator.next()

        sir = StatementIR(name)
        self.cache[name] = sir
        return sir

    def update(self, stmt_ir):
        name = stmt_ir.name
        self.cache[name] = stmt_ir

    def clear(self):
        want_clear = [
            key
            for key in self.cache.keys()
            if self.name_generator.match_name(key)
        ]
        for key in want_clear:
            del self.cache[key]
