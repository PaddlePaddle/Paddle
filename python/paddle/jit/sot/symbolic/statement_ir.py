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

"""
THIS FILE IS PRIVATE !!

use interface in symbolic_context.py first.
"""
from __future__ import annotations

import weakref
from typing import Any, Callable

from paddle.utils import is_sequence, map_structure

from ..utils import NameGenerator, OrderedSet, Singleton, flatten_extend


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
        stacks: list[str],
    ):
        assert type in ["call", "api", "method", "layer"]
        self.name = name
        self.inputs = inputs  # (list of Symbols, dict of Symbols)
        self.outputs = outputs  # list of Symbol | PythonObj
        self.stmt_stack = (
            stacks  # a list of string to record the source code callstack.
        )
        self.type = type

    def __str__(self):
        def to_string(inps):
            if isinstance(inps, str) or not is_sequence(inps):
                return inps.__str__()
            inps = (x.__str__() for x in inps)
            return ", ".join(inps)

        return "{} || {} = {} ({}) ".format(
            self.type + " " * (10 - len(self.type)),
            to_string(self.outputs),
            self.name,
            to_string(self.inputs),
        )

    def __repr__(self):
        return self.__str__()


class CallStatement(Statement):
    def __init__(
        self,
        name: str,
        inputs: list[Symbol],
        outputs: list[Symbol],
        stacks: list[str],
    ):
        super().__init__("call", name, inputs, outputs, stacks)
        self.sir_name = name


class ApiStatement(Statement):
    def __init__(
        self,
        api: Callable,
        inputs: list[Symbol],
        outputs: list[Symbol],
        stacks: list[str],
    ):
        super().__init__(
            "api", "paddle." + api.__name__, inputs, outputs, stacks
        )
        self.api = api


class MethodStatement(Statement):
    def __init__(
        self,
        name: str,
        inputs: list[Symbol],
        outputs: list[Symbol],
        stacks: list[str],
    ):
        super().__init__("method", name, inputs, outputs, stacks)
        self.method = name


class LayerStatement(Statement):
    def __init__(
        self,
        layer: Reference,  # Reference of paddle.nn.Layer
        inputs: list[Symbol],
        outputs: list[Symbol],
        stacks: list[str],
    ):
        super().__init__(
            "layer", layer.__class__.__name__, inputs, outputs, stacks
        )
        self.layer = layer


class StatementIR:
    """
    StatementIR is the carrier that records the code for building the neural network model.It is
    a representation of a purely computational structure, and does not care about specific values.
    The function converted from StatementIR can ensure that it can be turned into a static state.
    In this way, we can reuse the original `to_static` function to realize the execution of the static graph.

    Note:
        Don't create by yourself, just use the StatementIRCache.get()
    """

    def __init__(self, name: str):
        self.name = name
        self.inputs = []  # list of Symbol | PythonObj
        self.outputs = []  # list of Symbol | PythonObj
        self.statements = []  # list of Statement

    def __len__(self):
        return len(self.statements)

    def __deepcopy__(self, memo=None):
        new_sir = StatementIR(self.name)
        new_sir.inputs = list(self.inputs)
        new_sir.outputs = list(self.outputs)
        new_sir.statements = list(self.statements)
        return new_sir

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

        input_symbols = sorted(used_symbols, key=lambda x: x.name)
        return input_symbols

    def __str__(self):
        strs = []
        strs.append("StatmentIR: %s" % self.name)
        strs.append(f"  inputs: {map_structure(lambda x: x.name, self.inputs)}")
        strs.append(
            f"  outputs: {map_structure(lambda x: x.name, self.outputs)}"
        )
        strs.append("  statements: ")
        for stmt in self.statements:
            strs.append(f"    {stmt}")
        return "\n".join(strs)

    def __repr__(self):
        return self.__str__()

    def graph_size(self):
        call_layers = [x for x in self.statements if x.type == "layer"]
        return len(self.statements) + len(call_layers)


@Singleton
class StatementIRFactory:
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


@Singleton
class SIRRuntimeCache:
    """
    It is used to cache the runtime information of the StatementIR.
    """

    def __init__(self):
        self.cache = {}
        #     { name : (inputs, outputs, free_vars) }
        #       inputs  : can be used when call_SIR, if free_vars exist
        #       outputs : used for generator new ProxyTensor output before fallback
        #       free_vars: (name, function)

    def __getitem__(self, key):
        return self.cache[key]

    def has_key(self, key: str) -> bool:
        """
        has_key is used to check whether the key is in the cache.
        """
        return key in self.cache.keys()

    def set_origin_inputs(self, key: str, inputs: Any):
        """
        Set Cache origin Inputs of the StatementIR
        """
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (inputs, val[1], val[2])
        else:
            self.cache[key] = (inputs, None, None)

    def set_origin_outputs(self, key: str, outputs: Any):
        """
        Set Cache origin outputs of the StatementIR
        """
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], outputs, val[2])
        else:
            self.cache[key] = (None, outputs, None)

    def set_free_vars(self, key: str, free_vars: Any):
        """
        Set Cache free variables of the StatementIR
        """
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], val[1], free_vars)
        else:
            self.cache[key] = (None, None, free_vars)

    def get_origin_inputs(self, key: str):
        """
        Get the origin inputs of the StatementIR.
        """
        if key in self.cache.keys():
            return self.cache[key][0]
        else:
            return None

    def get_origin_outputs(self, key: str):
        """
        Get the origin outputs of the StatementIR.
        """
        if key in self.cache.keys():
            return self.cache[key][1]
        else:
            return None

    def get_free_vars(self, key: str):
        """
        Get the free variables of the StatementIR.
        """
        if key in self.cache.keys():
            return self.cache[key][2]
        else:
            return None
