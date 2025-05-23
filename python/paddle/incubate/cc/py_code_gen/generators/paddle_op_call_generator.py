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

import os
import sys

import paddle.incubate.cc.py_code_gen.util.lambda_util as fn
from paddle.incubate.cc.py_code_gen.generators.paddle_c_ops_arg_names import (
    GetCOpsArgNames,
    GetTypeNameByArgName,
)
from paddle.incubate.cc.py_code_gen.ir import ir_attr, ir_type


class GSOutputDimGenerator:

    def __init__(self, module_name, inputs, symbol_bindings):
        self.m = module_name
        self.symbol2py_code = {
            symbol_name: self._GenerateSymbolInputExpr(
                tag, inputs, tensor_idx, dim_idx
            )
            for tag, symbol_name, tensor_idx, dim_idx in self._GetSymbolBindings(
                inputs, symbol_bindings
            )
        }

    def Generate(self, dim_expr_attr):
        method_name = f"_Generate{type(dim_expr_attr).__name__}"
        return getattr(self, method_name)(dim_expr_attr)

    def _GenerateInt64Attribute(self, dim_expr_attr):
        return f"{dim_expr_attr.value}"

    def _GenerateStrAttribute(self, dim_expr_attr):
        return self.symbol2py_code[dim_expr_attr.value]

    def _GenerateArrayAttribute(self, dim_expr_attr):
        method_name = f"_Generate{dim_expr_attr.value[0].value}"
        return getattr(self, method_name)(*dim_expr_attr.value)

    def _GenerateNegative(self, _, operand):
        raise f"(- {self.Generate(operand)})"

    def _GenerateReciprocal(self, _, operand):
        raise NotImplementedError("dead code")

    def _GenerateAdd(self, _, *args):
        def IsNegative(operand):
            return (
                type(operand) is ir_attr.ArrayAttribute
                and operand.value[0].value == "Negative"
            )

        def ConvertOperand(i, operand):
            if IsNegative(operand):
                return (" - " if i > 0 else "-") + self.Generate(
                    operand.value[1]
                )
            else:
                return (" + " if i > 0 else "") + self.Generate(operand)

        ret = "".join(
            [ConvertOperand(i, operand) for i, operand in enumerate(args)]
        )
        return f"({ret})"

    def _GenerateMul(self, _, *args):
        def IsReciprocal(operand):
            return (
                type(operand) is ir_attr.ArrayAttribute
                and operand.value[0].value == "Reciprocal"
            )

        def ConvertOperand(i, operand):
            if IsReciprocal(operand):
                return (" / " if i > 0 else "1 / ") + self.Generate(
                    operand.value[1]
                )
            else:
                return (" * " if i > 0 else "") + self.Generate(operand)

        ret = "".join(
            [ConvertOperand(i, operand) for i, operand in enumerate(args)]
        )
        return f"({ret})"

    def _GenerateMax(self, _, arg0, *args):
        ret = self.Generate(arg0)
        for arg in args:
            ret = f"{self.m}.maximum({ret}, {self.Generate(arg)})"
        return ret

    def _GenerateMin(self, _, arg0, *args):
        ret = self.Generate(arg0)
        for arg in args:
            ret = f"{self.m}.minimum({ret}, {self.Generate(arg)})"
        return ret

    def _GenerateBroadcast(self, _, *args):
        empties = [f"{self.m}.empty({self.Generate(arg)})" for arg in args]
        empties_sum = " + ".join(empties)
        return f"{self.m}.shape({empties_sum})"

    def _GenerateSymbolInputExpr(self, tag, inputs, tensor_idx, dim_idx):
        method_name = f"_{tag}"
        return getattr(self, method_name)(inputs, tensor_idx, dim_idx)

    def _ShapeSymbolBinding(self, inputs, tensor_idx, dim_idx):
        return f"{inputs[tensor_idx].name}.shape[{dim_idx}]"

    def _DataSymbolBinding(self, inputs, tensor_idx, dim_idx):
        return f"{inputs[tensor_idx].name}[{dim_idx}]"

    def _GetSymbolBindings(self, inputs, symbol_bindings):
        for binding in symbol_bindings.value:
            tag, symbol_name, tensor_idx, dim_idx = binding.value
            yield tag.value, symbol_name.value, tensor_idx.value, dim_idx.value


class CinnOpCallGenerator:

    def cinn_op_yield_store(self, op, x):
        return lambda f: f"{f(x.name)},"

    def cinn_op_concat(self, op, *inputs):
        operands = fn.join_map([input.name for input in inputs])
        return (
            lambda f: f"{self.m}.concat([{operands(f)}], axis={op.attrs['axis']})"
        )

    def cinn_op_slice(self, op, x):
        return (
            lambda f: f"{self.m}.slice({f(x.name)}, axes={op.attrs['axes']}, starts={op.attrs['starts']}, ends={op.attrs['ends']})"
        )

    def cinn_op_reduce_sum(self, op, x):
        return (
            lambda f: f"{self.m}.sum({f(x.name)}, keepdim={op.attrs['keep_dim']}, axis={op.attrs['dim']})"
        )

    def cinn_op_reduce_prod(self, op, x):
        return (
            lambda f: f"{self.m}.prod({f(x.name)}, keepdim={op.attrs['keep_dim']}, axis={op.attrs['dim']})"
        )

    def cinn_op_scale(self, op, x):
        return (
            lambda f: f"{f(x.name)} * {op.attrs['scale']} + {op.attrs['bias']}"
        )

    def cinn_op_broadcast(self, op, x):
        return (
            lambda f: f"{self.m}.broadcast_to({f(x.name)}, {op.attrs['out_shape']})"
        )

    def cinn_op_reshape(self, op, x):
        return lambda f: f"{self.m}.reshape({f(x.name)}, {op.attrs['shape']})"

    def cinn_op_generate_shape(self, op, *inputs):
        generator = GSOutputDimGenerator(
            self.m,
            inputs,
            symbol_bindings=op.base_op.attrs["symbol_bindings"],
        )
        output_dims = ", ".join(
            [
                generator.Generate(dim_expr)
                for dim_expr in op.base_op.attrs["output_dim_exprs"].value
            ]
        )
        input_args = fn.join_map([input.name for input in inputs])
        return lambda f: f"[{output_dims}] # inputs: {input_args(f)}"


class PaddleOpCallGenerator(CinnOpCallGenerator):

    def __init__(self, module_name="paddle"):
        self.m = module_name

    def let(self, args_value_pair, body):
        args, value = args_value_pair
        assert callable(value)
        assert callable(body)
        if isinstance(args, (list, tuple)):
            args = ", ".join(args)
        return (
            lambda f: f"(lambda x, f: f(x))({value(f)}, lambda {args}: {body(f)})"
        )

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
                fn.const(attrs[arg_name])
                if arg_name in attrs
                else GetPosArgVarName(arg_name)
            )
            for arg_name in arg_names
        ]
        args_str = fn.join_map(args)

        def GetMethodName():
            inplace = (op_name is not None) and (
                self.PaddleMethodName(op) == f"{op_name}_"
            )
            enable_fallback_from_inplace = self.EnableFallbackFromInplace()
            if inplace and not enable_fallback_from_inplace:
                return f"{op_name}_"
            return op_name

        method_name = GetMethodName()
        out = lambda f: f"{m}.{method_name}({args_str(f)})"
        if len(op.output_types) <= 1:
            return out
        nones = ",".join("None" for i in range(len(op.output_types) - 1))
        return self.let(
            ("out", out),
            lambda f: f"out if isinstance(out, (list, tuple)) else (out, {nones})",
        )

    def EnableFallbackFromInplace(self):
        return os.getenv("ATHENA_ENABLE_FALLBACK_FROM_INPLACE") not in {
            "0",
            "False",
            "false",
            "OFF",
            "off",
        }

    def pd_op_arange(self, op, start, end, stop):
        start_name = start.name
        end_name = end.name
        stop_name = stop.name
        dtype = op.attrs["dtype"].split(".")[-1]
        return (
            lambda f: f"{self.m}.arange({f(start_name)}, {f(end_name)}, {f(stop_name)}, dtype='{dtype}')"
        )

    def pd_op_one_hot(self, op, x, num_classes):
        x_name = x.name
        num_classes_name = num_classes.name
        N = lambda f: f"{self.m}.cast({f(num_classes_name)}, {f(x_name)}.dtype)"
        return (
            lambda f: f"{self.m}._C_ops.one_hot({f(x_name)} % {N(f)}, {f(num_classes_name)})"
        )

    def pd_op_assign(self, op, x):
        return lambda f: f(x.name)

    def pd_op_assign_value(self, op):
        shape = op.attrs["shape"]
        values = op.attrs["values"]
        dtype = op.attrs["dtype"]
        return (
            lambda f: f"{self.m}.to_tensor({values}, dtype={dtype}).reshape({shape})"
        )

    def pd_op_shape(self, op, input):
        input_name = input.name
        if isinstance(input.dtype, ir_type.Float16Type):
            input = lambda f: f"{self.m}.cast({f(input_name)}, 'float32')"
        return self.GenerateCOpsCall(op, [input])

    def cf_yield(self, op, *inputs):
        return None

    def builtin_shadow_output(self, op, *inputs):
        return None

    def builtin_parameter(self, op):
        return None

    def builtin_constant(self, op):
        return None

    def pd_op_data(self, op):
        return None

    def pd_op_feed(self, op):
        return None

    def builtin_combine(self, op, *inputs):
        input_names = [input.name for input in inputs]
        return (
            lambda f: f"[{', '.join(f(input_name) for input_name in input_names)}]"
        )

    def builtin_slice(self, op, x):
        index = op.attrs["index"]
        return lambda f: f"{f(x.name)}[{index}]"

    def builtin_split(self, op, x):
        return lambda f: f(x.name)

    def pd_op_set_value_(self, op, x, starts, ends, steps):
        return (
            lambda f: f"{self.m}._C_ops.set_value_({f(x.name)}, {f(starts.name)}, {f(ends.name)}, {f(steps.name)}, {op.attrs['axes']}, {op.attrs['decrease_axes']}, {op.attrs['none_axes']}, {op.attrs['shape']}, {op.attrs['values']})"
        )

    def pd_op_share_data_(self, op, x):
        return lambda f: f"{f(x.name)}.detach()"

    def pd_op_full_int_array(self, op):
        return lambda f: op.attrs["value"]

    def pd_op_batch_norm_(self, op, *inputs):
        return self.GenerateCOpsCall(op, inputs, op_name="batch_norm")

    def pd_op_rnn_(self, op, *inputs):
        outs = self.GenerateCOpsCall(op, inputs, op_name="rnn")
        return lambda f: f"{outs(f)} + (None,)"

    def pd_op_select_input(self, op, cond, elem0, elem1):
        return (
            lambda f: f"({f(elem0.name)} if {f(cond.name)} == 0 else {f(elem1.name)})"
        )
