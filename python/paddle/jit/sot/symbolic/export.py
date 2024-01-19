# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
from itertools import chain

import paddle
from paddle.utils import flatten

from ..utils import ConstTypes, ExportError, NameGenerator
from .statement_ir import Symbol


class PyStatement:
    tab = " " * 4

    def __init__(self, *lines):
        self.sub_statement = []
        self.lines = lines

    def get_lines(self, prefix=""):
        lines = [prefix + line for line in self.lines]
        for statment in self.sub_statement:
            lines.extend(statment.get_lines(self.tab + prefix))
        return lines

    def add_sub(self, *lines):
        sub = PyStatement(*lines)
        self.sub_statement.append(sub)
        return sub

    def __str__(self):
        return "\n".join(self.get_lines())


class NameGener:
    def __init__(self, SIR):
        self.SIR = SIR
        self.name_map = {}
        self.param_name_generator = NameGenerator("self.parameter_")
        self.non_param_name_generator = NameGenerator("var_")

    def __call__(self, var):
        if isinstance(var, Symbol):
            if var not in self.name_map:
                self.register_symbol(var)
            return self.name_map[var]

        elif isinstance(var, str):
            return f"'{var}'"
        else:
            return str(var)

    def register_symbol(self, symbol):
        if symbol in self.SIR.param_symbol:
            name = self.param_name_generator.next()
        else:
            name = self.non_param_name_generator.next()
        self.name_map[symbol] = name


class PyFileGen:
    def __init__(self, SIR):
        self.SIR = SIR
        self.roots = []

        self.name_gener = NameGener(self.SIR)

        self.SIR_sig = "||".join(
            f"{stmt.type}:{stmt.name}" for stmt in SIR.statements
        )

    def new_root(self, *args):
        stmt = PyStatement(*args)
        self.roots.append(stmt)
        return stmt

    def roots_to_string(self):
        lines = []
        for root in self.roots:
            lines.extend(root.get_lines())
        return "\n".join(lines)

    def gen_py_codes(self):
        self.check_exportable()
        self.create_header()
        self.new_root("\n")
        self.create_layer()
        self.new_root("\n")
        self.create_test()
        self.new_root("\n")
        self.create_tail()
        return self.roots_to_string()

    def check_exportable(self):
        for stmt in self.SIR.statements:
            for inp in flatten(stmt.inputs):
                if not isinstance(inp, ConstTypes) and not isinstance(
                    inp, Symbol
                ):
                    raise ExportError(
                        f"Not support create python file with input: {inp}"
                    )
            for out in flatten(stmt.outputs):
                if not isinstance(out, ConstTypes) and not isinstance(
                    out, Symbol
                ):
                    raise ExportError(
                        f"Not support create python file with output: {out}"
                    )

    def create_header(self):
        self.new_root(
            f"# {self.SIR_sig}",
            "import paddle",
            "import unittest",
            "import numpy as np",
        )

    def create_layer(self):
        layer_class = self.new_root("class LayerCase(paddle.nn.Layer):")

        init_fn = layer_class.add_sub("def __init__(self):")
        init_fn.add_sub("super().__init__()")

        for param in self.SIR.param_symbol:
            meta = self.SIR.symbol_meta_map[param]
            init_fn.add_sub(
                f"{self.name_gener(param)} = self.create_parameter(",
                f"   shape={meta.shape},",
                f"   dtype={meta.dtype},",
                ")",
            )

        for stmt in self.SIR.statements:
            if stmt.type == "layer":
                layer = stmt.layer()
                init_fn.add_sub(self.init_sub_layer(layer))

        forward_definition = ["def forward(", "    self,"]

        for inp in self.SIR.inputs:
            if inp in self.SIR.non_param_symbol:
                meta = self.SIR.symbol_meta_map[inp]
                forward_definition.append(
                    f"    {self.name_gener(inp)},    # {str(meta)}"
                )
        forward_definition.append("):")

        forward_fn = layer_class.add_sub(*forward_definition)

        for stmt in self.SIR.statements:
            forward_fn.add_sub(*self.create_stmt_line(stmt))

        forward_fn.add_sub(
            "return {}".format(
                ", ".join(self.name_gener(out) for out in self.SIR.outputs)
            )
        )

    def create_test(self):
        test_class = self.new_root("class TestLayer(unittest.TestCase):")

        setup = test_class.add_sub("def setUp(self):")
        test_inputs = [
            "self.inputs = (",
        ]
        for inp in self.SIR.inputs:
            if inp in self.SIR.non_param_symbol:
                meta = self.SIR.symbol_meta_map[inp.name]
                shape_str = "[1]" if len(meta.shape) == 0 else str(meta.shape)
                if meta.dtype in (
                    paddle.int8,
                    paddle.int16,
                    paddle.int32,
                    paddle.int64,
                ):
                    test_inputs.append(
                        f"    paddle.randint(low=0, high=10, shape={shape_str}, dtype={meta.dtype}),"
                    )
                else:
                    test_inputs.append(
                        f"    paddle.rand(shape={shape_str}, dtype={meta.dtype}),"
                    )
        test_inputs.append(")")
        setup.add_sub(*test_inputs)
        setup.add_sub("self.net = LayerCase()")

        train = test_class.add_sub(
            "def train(self, net, to_static, with_prim=False, with_cinn=False):"
        )
        train.add_sub(
            "paddle.set_flags({'FLAGS_prim_all': with_prim})",
            "if to_static:",
            "    if with_cinn:",
            "        build_strategy = paddle.static.BuildStrategy()",
            "        build_strategy.build_cinn_pass = True",
            "        net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)",
            "    else:",
            "        net = paddle.jit.to_static(net, full_graph=True)",
            "outs = net(*self.inputs)",
            "return outs",
        )

        test_ast_cinn_static = test_class.add_sub(
            "def test_ast_prim_cinn(self):"
        )
        test_ast_cinn_static.add_sub(
            "st_out = self.train(self.net, to_static=True)",
            "cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)",
            "for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):",
            "    np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)",
        )

    def create_tail(self):
        self.new_root(
            "if __name__ == '__main__':",
            "    unittest.main()",
        )

    def init_sub_layer(self, layer, layer_name):
        # TODO @wuzhanfei need more effecient way to create a sub layer
        # now, we just close call_Layer behavior
        raise ExportError("Not support create sub layer now.")

    def create_input_string(self, args, kwargs):
        return ", ".join(
            chain(
                (self.name_gener(arg) for arg in args),
                (f"{k}={self.name_gener(v)}" for k, v in kwargs.items()),
            )
        )

    def create_unpack_output_string(self, outputs):
        path = ["out"]
        result = []

        def search(outputs, path, result):
            if isinstance(outputs, (list, tuple)):
                search_sequnce(outputs, path, result)
            elif isinstance(outputs, dict):
                search_dict(outputs, path, result)
            elif isinstance(outputs, Symbol):
                result.append(self.name_gener(outputs) + " = " + "".join(path))

        def search_sequnce(outputs, path, result):
            for idx, out in enumerate(outputs):
                path.append(f"[{idx}]")
                search(out, path, result)
                path.pop()

        def search_dict(outputs, path, result):
            for k, out in outputs.items():
                path.append(f"[{k}]")
                search(out, path, result)
                path.pop()

        search(outputs, path, result)
        return result

    def create_stmt_line(self, stmt):
        return getattr(self, "create_" + stmt.type + "_stmt")(stmt)

    def create_api_stmt(self, stmt):
        args, kwargs = stmt.inputs
        input_str = self.create_input_string(args, kwargs)
        api = stmt.api
        api_str = api.__module__ + "." + api.__name__
        if isinstance(stmt.outputs, Symbol):
            return [f"{self.name_gener(stmt.outputs)} = {api_str}({input_str})"]
        else:
            compute_code = f"out = {api_str}({input_str})"
            unpack_codes = self.create_unpack_output_string(stmt.outputs)
            return [compute_code] + unpack_codes

    def create_method_stmt(self, stmt):
        args, kwargs = stmt.inputs
        input_str = self.create_input_string(args[1:], kwargs)
        method_str = self.name_gener(args[0]) + "." + stmt.method
        if isinstance(stmt.outputs, Symbol):
            return [
                f"{self.name_gener(stmt.outputs)} = {method_str}({input_str})"
            ]
        else:
            compute_code = f"out = {method_str}({input_str})"
            unpack_codes = self.create_unpack_output_string(stmt.outputs)
            return [compute_code] + unpack_codes


def export(SIR, path):
    try:
        pygen = PyFileGen(SIR)
        string = pygen.gen_py_codes()
    except ExportError as e:
        print(f"[SOT] Export {SIR.name} Failed:", e)
        return

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f"{SIR.name}.py"), "w") as f:
        f.write(string)
        print(f"[SOT] Export {SIR.name} Sucess with size {len(SIR.statements)}")
