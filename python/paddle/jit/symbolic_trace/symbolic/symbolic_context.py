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

import paddle

from ..utils import NameGenerator, log
from .compile_cache import CompileSIRCache
from .statement_ir import Statement, StatementIR, StatementIRFactory, Symbol


class SymbolicTraceContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.statement_factory = StatementIRFactory()
        self.statement_factory.clear()
        self.sir_stack = [self.statement_factory.create()]
        self.layer_name_generator = NameGenerator("layer_")

    @property
    def TOS(self):
        return self.sir_stack[-1]

    def new_layername(self):
        return self.layer_name_generator.next()

    def call_SIR(self, sirname, inputs, outputs):
        stmt = Statement("call", sirname, inputs, outputs)
        self.TOS.add_statement(stmt)

    def call_API(self, api, inputs, outputs):
        assert callable(api), "call_API must receive a paddle api."
        stmt = Statement("api", api, inputs, outputs)
        self.TOS.add_statement(stmt)

    def call_METHOD(self, method_name, inputs, outputs):
        assert isinstance(
            method_name, str
        ), "call_METHOD must method api name. string."
        assert isinstance(
            inputs[0][0], Symbol
        ), "call_METHOD must first augument must be Symbol Variable."
        stmt = Statement("method", method_name, inputs, outputs)
        self.TOS.add_statement(stmt)

    def call_LAYER(self, layer_name, inputs, outputs):
        stmt = Statement("layer", layer_name, inputs, outputs)
        self.TOS.add_statement(stmt)

    def get_sir(self, name):
        return self.statement_factory[name]

    def reset_TOS(self):
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    def replace_TOS(self, sir):
        """Use deepcopyed sir to replace the TOS.
        This function will update statment_factory.
        """
        self.sir_stack.pop()
        self.sir_stack.append(sir)
        self.statement_factory.update(sir)

    def compile_do_nothing(self, ret_vals):
        def dummy_func(*args, **kwargs):
            return []

        # return None function
        dummy_stmt_ir = StatementIR("dummy_func")
        dummy_stmt_ir.outputs = []
        dummy_stmt_ir.inputs = []
        return dummy_func, dummy_stmt_ir

    def compile_fn(self, ret_vals):
        """
        start compile and return the python function, which must can be to_static without errors.
        """
        cur_sir: StatementIR = self.TOS
        # step0: if no statement, return a dummy function
        if len(cur_sir.statements) == 0:
            return self.compile_do_nothing(ret_vals)
        # step1: analyse sir inputs and outputs
        cur_sir.inputs = cur_sir.analyse_inputs()
        # TODO: output analysis
        cur_sir.outputs = paddle.utils.map_structure(
            lambda x: Symbol(x.name), ret_vals
        )
        log(1, "start subgraph compile and execution.\n")
        log(1, self.TOS, "\n")
        # step2: call compile_sir and get python function, third cache is triggered here.
        static_func = CompileSIRCache()(self, cur_sir.name)
        # step3: GC and reset TOS
        # self.reset_TOS()

        return static_func, cur_sir
