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

from copy import deepcopy

from paddle.utils import flatten, is_sequence, map_structure

from ..utils import NameGenerator, Singleton


class Symbol:
    """
    we need this class to distinguish the string and `math variable`
    """

    def __init__(self, name):
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
    def __init__(self, type, name, inputs, outputs):
        assert type in ["call", "api", "method", "layer"]
        self.name = name
        self.inputs = inputs  # (list of Symbols, dict of Symbols)
        self.outputs = outputs  # list of Symbol | PythonObj
        self.type = type

    def __deepcopy__(self, memo=None):
        return Statement(
            self.type, self.name, deepcopy(self.inputs), deepcopy(self.outputs)
        )

    def __str__(self):
        def to_string(inps):
            if isinstance(inps, str) or not is_sequence(inps):
                return inps.__str__()
            inps = (x.__str__() for x in inps)
            return ", ".join(inps)

        name = (
            self.name
            if isinstance(self.name, str)
            else "paddle." + self.name.__name__
        )
        return "{} || {} = {} ({}) ".format(
            self.type + " " * (10 - len(self.type)),
            to_string(self.outputs),
            name,
            to_string(self.inputs),
        )

    def __repr__(self):
        return self.__str__()


class StatementIR:
    """
    Don't create by yourself, just use the StatementIRCache.get()
    """

    def __init__(self, name):
        self.name = name
        self.inputs = []  # list of Symbol | PythonObj
        self.outputs = []  # list of Symbol | PythonObj
        self.statements = []  # list of Statement

    def __deepcopy__(self, memo=None):
        new_sir = StatementIR(self.name)
        new_sir.inputs = deepcopy(self.inputs)
        new_sir.outputs = deepcopy(self.outputs)
        new_sir.statements = deepcopy(self.statements)
        return new_sir

    def add_input(self, input):
        self.inputs.append(input)

    def add_output(self, output):
        self.outputs.append(output)

    def add_statement(self, statement):
        assert isinstance(statement, Statement)
        self.statements.append(statement)

    def analyse_inputs(self):
        used_symbols = set()
        generated_symbols = set()
        for stmt in self.statements:
            for inp in flatten(stmt.inputs):
                if isinstance(inp, Symbol):
                    used_symbols.add(inp)
            for out in flatten(stmt.outputs):
                if isinstance(out, Symbol):
                    generated_symbols.add(out)

        input_symbols = list(used_symbols - generated_symbols)
        input_symbols = sorted(input_symbols, key=lambda x: x.name)
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


@Singleton
class StatementIRFactory:
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
    def __init__(self):
        self.cache = {}
        #       { name : (inputs, outputs, free_vars) }
        #       inputs  : can be used when call_SIR, if free_vars exist
        #       outputs : used for generator new ProxyTensor output before fallback
        #       free_vars: (name, function)

    def __getitem__(self, key):
        return self.cache[key]

    def has_key(self, key):
        return key in self.cache.keys()

    def set_origin_inputs(self, key, inputs):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (inputs, val[1], val[2])
        else:
            self.cache[key] = (inputs, None, None)

    def set_origin_outputs(self, key, outputs):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], outputs, val[2])
        else:
            self.cache[key] = (None, outputs, None)

    def set_free_vars(self, key, free_vars):
        if key in self.cache.keys():
            val = self.cache[key]
            self.cache[key] = (val[0], val[1], free_vars)
        else:
            self.cache[key] = (None, None, free_vars)

    def get_origin_inputs(self, key):
        if key in self.cache.keys():
            return self.cache[key][0]
        else:
            return None

    def get_origin_outputs(self, key):
        if key in self.cache.keys():
            return self.cache[key][1]
        else:
            return None

    def get_free_vars(self, key):
        if key in self.cache.keys():
            return self.cache[key][2]
        else:
            return None
