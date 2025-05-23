# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import paddle.incubate.cc.py_code_gen.util.lambda_util as fn
from paddle.incubate.cc.py_code_gen.generators.paddle_c_ops_arg_names import (
    GetCOpsArgNames,
    GetTypeNameByArgName,
)
from paddle.incubate.cc.py_code_gen.ir import ir_type


class ApVariadicOpCallGenerator:

    def __init__(
        self,
        literal_str2local_id,
        lambda_str_attr_getter_name,
        module_name="paddle",
    ):
        self.literal_str2local_id = literal_str2local_id
        self.lambda_str_attr_getter_name = lambda_str_attr_getter_name
        self.m = module_name

    def GenerateOpCall(self, op, *inputs):
        method_name = op.GetPyVarName()
        if hasattr(self, method_name):
            return getattr(self, method_name)(op, *inputs)
        return self._GenerateOpCall(op, *inputs)

    def _GenerateOpCall(self, op, *inputs):
        assert op.GetValidPyVarNameComponents()[0] == "pd_op"

        def GetCOpsArgNamesThenCheck(op_name, attrs):
            c_ops_arg_names = GetCOpsArgNames(op_name)
            if c_ops_arg_names is None:
                return None
            attr_names = [
                attr_name for attr_name in c_ops_arg_names if attr_name in attrs
            ]
            assert len(inputs) + len(attr_names) == len(
                c_ops_arg_names
            ), f"op: {op.name}, len(inputs): {len(inputs)}, attr_names: {attr_names}, c_ops_arg_names: {c_ops_arg_names}"
            return c_ops_arg_names

        op_name = self.PaddleMethodName(op)
        c_ops_arg_names = GetCOpsArgNamesThenCheck(op_name, op.attrs)
        if c_ops_arg_names is not None:
            return self.GenerateCOpsCall(op, inputs=inputs)
        print(
            f"c_ops_arg_names found. op: {op.name}, op: {op}", file=sys.stderr
        )
        return lambda f: f"paddle.{op_name}('{op.name} not found.')"

    def PaddleMethodName(self, op):
        return op.GetValidPyVarNameComponents()[-1]

    def GetOpAttrs(self, op):
        ignored_attr_names = {
            "__operands_symbols_signature__",
            "__results_symbols_signature__",
            "stop_gradient",
            "place",
        }
        for attr_name, attr_value in op.attrs.items():
            if attr_name in ignored_attr_names:
                continue
            yield attr_name, attr_value

    def GenerateCOpsCall(self, op, inputs, op_name=None):
        attrs = op.attrs
        op_name = op_name if op_name is not None else self.PaddleMethodName(op)
        arg_names = GetCOpsArgNames(op_name)
        pos_arg_idx = -1

        def GetPosArgVarName(arg_name):
            nonlocal pos_arg_idx
            pos_arg_idx += 1
            t = inputs[pos_arg_idx]
            assert not isinstance(t, str)
            if callable(t):
                return t
            if t is None:
                return lambda f: "None"
            if isinstance(t.type, ir_type.NullType):
                return lambda f: "None"
            type_name = GetTypeNameByArgName(op_name, arg_name)
            t_name = t.name
            if type_name == "IntArray" and isinstance(
                t.type, ir_type.VectorType
            ):
                return lambda f: f"[x.reshape([]) for x in {f(t_name)}]"
            return lambda f: f(t_name)

        m = f"{self.m}._C_ops"
        args = [
            (
                fn.const(self._convert_attr_pycode(arg_name, attrs[arg_name]))
                if arg_name in attrs
                else GetPosArgVarName(arg_name)
            )
            for arg_name in arg_names
        ]
        args_str = fn.join_map(args)

        def GetMethodName():
            return op_name

        method_name = GetMethodName()
        out = lambda f: f"{m}.{method_name}({args_str(f)})"
        assert len(op.output_types) == 1
        return out

    def _convert_attr_pycode(self, attr_name, attr_value):
        if attr_name in (
            "code_module_lambda",
            "infer_symbolic_lambda",
            "infer_meta_lambda",
            "kernel_dispatch_lambda",
            "kernel_dispatch_const_data_lambda",
        ):
            return self._convert_lambda_str_attr_pycode(attr_value)
        else:
            return attr_value

    def _convert_lambda_str_attr_pycode(self, attr_value):
        local_id = len(self.literal_str2local_id)
        self.literal_str2local_id[attr_value] = local_id
        return f"self.{self.lambda_str_attr_getter_name}({local_id})"
