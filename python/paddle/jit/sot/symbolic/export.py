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


class PyFileGen:
    def __init__(self, SIR):
        self.SIR = SIR
        self.roots = []

        self.layer_name_map = {}
        self.layer_name_generator = NameGenerator("_")
        self.SIR_name = SIR.name.replace("_", "")

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
            "import paddle",
            "import unittest",
            "import numpy as np",
        )

    def create_layer(self):
        layer_class = self.new_root(f"class {self.SIR_name}(paddle.nn.Layer):")

        init_fn = layer_class.add_sub("def __init__(self):")
        init_fn.add_sub("super().__init__()")

        for param in self.SIR.param_symbol:
            meta = self.SIR.symbol_meta_map[param.name]
            init_fn.add_sub(
                f"self.{param.name} = self.create_parameter(",
                f"   shape={meta.shape},",
                f"   dtype={meta.dtype},",
                ")",
            )

        for stmt in self.SIR.statements:
            if stmt.type == "layer":
                layer = stmt.layer()
                if id(layer) not in self.layer_name_map:
                    layer_name = (
                        layer.__class__.__name__
                        + self.layer_name_generator.next()
                    )
                    self.layer_name_map[id(layer)] = layer_name
                    init_fn.add_sub(self.init_sub_layer(layer, layer_name))

        forward_definition = ["def forward(", "    self,"]

        for inp in self.SIR.inputs:
            if inp in self.SIR.non_param_symbol:
                meta = self.SIR.symbol_meta_map[inp.name]
                forward_definition.append(f"    {inp.name},    # {str(meta)}")
        forward_definition.append("):")

        forward_fn = layer_class.add_sub(*forward_definition)

        for stmt in self.SIR.statements:
            forward_fn.add_sub(*self.create_stmt_line(stmt))

        forward_fn.add_sub(
            "return {}".format(
                ", ".join(self.true_name(out) for out in self.SIR.outputs)
            )
        )

    def create_test(self):
        test_class = self.new_root(
            f"class Test{self.SIR_name}(unittest.TestCase):"
        )

        setup = test_class.add_sub("def setUp(self):")
        test_inputs = [
            "self.inputs = (",
        ]
        for inp in self.SIR.inputs:
            if inp in self.SIR.non_param_symbol:
                meta = self.SIR.symbol_meta_map[inp.name]
                test_inputs.append(
                    f"    paddle.rand(shape={meta.shape}, dtype={meta.dtype}),"
                )
        test_inputs.append(")")
        setup.add_sub(*test_inputs)

        train = test_class.add_sub(
            "def train(self, net, to_static, with_cinn=False):"
        )
        train.add_sub(
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

        test_ast_static = test_class.add_sub("def test_ast_static(self):")
        test_ast_static.add_sub(
            "net = SIR0()",
            "dy_out = self.train(net, to_static=False)",
            "st_out = self.train(net, to_static=True, with_cinn=False)",
            "for dy, st in zip(paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)):",
            "    np.testing.assert_allclose(dy.numpy(), st.numpy(), atol=1e-8)",
        )

        test_ast_cinn_static = test_class.add_sub(
            "def test_ast_cinn_static(self):"
        )
        test_ast_cinn_static.add_sub(
            "net = SIR0()",
            "dy_out = self.train(net, to_static=False)",
            "st_out = self.train(net, to_static=True, with_cinn=True)",
            "for dy, st in zip(paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)):",
            "    np.testing.assert_allclose(dy.numpy(), st.numpy(), atol=1e-8)",
        )

    def create_tail(self):
        self.new_root(
            "if __name__ == '__main__':",
            "    unittest.main()",
        )

    def true_name(self, var):
        if isinstance(var, Symbol):
            if var in self.SIR.param_symbol:
                return "self." + var.name
            else:
                return var.name
        else:
            return str(var)

    def init_sub_layer(self, layer, layer_name):
        # TODO @wuzhanfei need more effecient way to create a sub layer
        # now, we just close call_Layer behavior
        raise ExportError("Not support create sub layer now.")

    def create_input_string(self, args, kwargs):
        return ", ".join(
            chain(
                (self.true_name(arg) for arg in args),
                (f"{k}={self.true_name(v)}" for k, v in kwargs.items()),
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
                result.append(self.true_name(outputs) + " = " + "".join(path))

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
            return [f"{stmt.outputs.name} = {api_str}({input_str})"]
        else:
            compute_code = f"out = {api_str}({input_str})"
            unpack_codes = self.create_unpack_output_string(stmt.outputs)
            return [compute_code] + unpack_codes

    def create_method_stmt(self, stmt):
        args, kwargs = stmt.inputs
        input_str = self.create_input_string(args[1:], kwargs)
        method_str = args[0].name + "." + stmt.method
        if isinstance(stmt.outputs, Symbol):
            return [f"{stmt.outputs.name} = {method_str}({input_str})"]
        else:
            compute_code = f"out = {method_str}({input_str})"
            unpack_codes = self.create_unpack_output_string(stmt.outputs)
            return [compute_code] + unpack_codes

    def create_layer_stmt(self, stmt):
        args, kwargs = stmt.inputs
        input_str = self.create_input_string(args, kwargs)
        layer_str = "self." + self.layer_name_map[id(stmt.layer())]
        if isinstance(stmt.outputs, Symbol):
            return [f"{stmt.outputs.name} = {layer_str}({input_str})"]
        else:
            compute_code = f"out = {layer_str}({input_str})"
            unpack_codes = self.create_unpack_output_string(stmt.outputs)
            return [compute_code] + unpack_codes


def export(SIR, path):
    try:
        pygen = PyFileGen(SIR)
        string = pygen.gen_py_codes()
    except ExportError as e:
        print("[SOT] Export SIR Failed:", e)
        return

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f"{SIR.name}.py"), "w") as f:
        f.write(string)
