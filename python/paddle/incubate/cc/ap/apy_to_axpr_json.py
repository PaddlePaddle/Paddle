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

from __future__ import annotations

import ast
import functools
import itertools
import json
import operator
import typing as t
from dataclasses import dataclass


def convert_python_stmts_to_axpr_json(python_code_stmts_str):
    tree = ast.parse(python_code_stmts_str)
    parser = PyToAnfParser()
    return parser(tree).ConvertToAnfExpr().JsonDump()


@dataclass
class AnfExpr:

    def DumpToFileAsJson(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.value, f, indent=2)

    def JsonDump(self):
        return json.dumps(self.value)


@dataclass
class AtomicAnfExpr(AnfExpr):
    value: t.Any


@dataclass
class CombinedAnfExpr(AnfExpr):
    value: t.Any


@dataclass
class AnfParseResult:
    bindings: list[str]
    body_atomic_anf_expr: AtomicAnfExpr

    def __add__(self, other):
        return AnfParseResult(
            bindings=[*self.bindings, *other.bindings],
            body_atomic_anf_expr=other.body_atomic_anf_expr,
        )

    def ConvertToAnfExpr(self):
        ret = self.body_atomic_anf_expr
        if len(self.bindings) == 0:
            return ret
        assert isinstance(ret, AtomicAnfExpr)
        ret = CombinedAnfExpr(
            ["__builtin_identity__", self.body_atomic_anf_expr.value]
        )
        return CombinedAnfExpr(["__builtin_let__", self.bindings, ret.value])


class PyToAnfParser:
    def __init__(self, seq_no_counter=None, return_count_constraint=None):
        self.bindings = []
        self.seq_no_counter = (
            seq_no_counter if seq_no_counter is not None else itertools.count()
        )
        self.return_count_constraint = (
            return_count_constraint
            if return_count_constraint is not None
            else ReturnCounterConstraint(limits=1)
        )

    def __call__(self, tree):
        ret = self.Parse(tree)
        return AnfParseResult(bindings=self.bindings, body_atomic_anf_expr=ret)

    def Parse(self, tree):
        method_name = f"Parse{type(tree).__name__}"
        return getattr(self, method_name)(tree)

    def ParseImport(self, tree):
        for alias in tree.names:
            assert isinstance(alias, ast.alias)
            name = alias.name
            asname = alias.asname if alias.asname is not None else name
            self.Bind(asname, ["import", {"str": name}])
        return AtomicAnfExpr(None)

    def ParseClassDef(self, tree: ast.ClassDef):
        assert len(tree.keywords) == 0
        class_name = tree.name

        def GetBases():
            bases = [self.Parse(base) for base in tree.bases]
            return self.BindToTmpVar(
                ['__builtin_list__', *[x.value for x in bases]]
            )

        def GetFunctions():
            body_name_and_method_pair = []
            for func_def in tree.body:
                if isinstance(func_def, ast.Pass):
                    continue
                assert isinstance(
                    func_def, ast.FunctionDef
                ), f"only method supported in class definition, {type(func_def)} were given."
                func_code = self.BindToTmpVar(
                    [
                        '__builtin_getattr__',
                        self.Parse(func_def).value,
                        {"str": '__function__'},
                    ]
                )
                pair = self.BindToTmpVar(
                    [
                        "__builtin_list__",
                        {"str": func_def.name},
                        func_code.value,
                    ]
                )
                body_name_and_method_pair.append(pair)
            positional_args = self.BindToTmpVar(['__builtin_list__'])
            keyword_args = self.BindToTmpVar(
                [
                    '__builtin_list__',
                    *[x.value for x in body_name_and_method_pair],
                ]
            )
            packed_args = self.BindToTmpVar(
                [
                    '__builtin_PackedArgs__',
                    positional_args.value,
                    keyword_args.value,
                ]
            )
            return self.BindToTmpVar(
                ['BuiltinSerializableAttrMap', packed_args.value]
            )

        class_anf_expr = self.BindToTmpVar(
            [
                'type',
                {"str": class_name},
                GetBases().value,
                GetFunctions().value,
            ]
        )
        for elt in reversed(tree.decorator_list):
            decorator = self.Parse(elt)
            class_anf_expr = self.BindToTmpVar(
                [decorator.value, class_anf_expr.value]
            )
        self.Bind(class_name, class_anf_expr)
        return class_anf_expr

    def Parsekeyword(self, tree):
        value = self.Parse(tree.value)
        return self.BindToTmpVar(
            ["__builtin_list__", {"str": tree.arg}, value.value]
        )

    def ParseBinOp(self, tree):
        left = self.Parse(tree.left)
        op = self.Parse(tree.op)
        right = self.Parse(tree.right)
        return self.BindToTmpVar([op.value, left.value, right.value])

    def ParseUnaryOp(self, tree):
        op = self.Parse(tree.op)
        operand = self.Parse(tree.operand)
        return self.BindToTmpVar([op.value, operand.value])

    def ParseCompare(self, tree):
        assert len(tree.ops) == 1
        op = self.Parse(tree.ops[0])
        left = self.Parse(tree.left)
        assert len(tree.comparators) == 1
        right = self.Parse(tree.comparators[0])
        return self.BindToTmpVar([op.value, left.value, right.value])

    def ParseAdd(self, tree):
        return AtomicAnfExpr("__builtin_Add__")

    def ParseSub(self, tree):
        return AtomicAnfExpr("__builtin_Sub__")

    def ParseMult(self, tree):
        return AtomicAnfExpr("__builtin_Mul__")

    def ParseDiv(self, tree):
        return AtomicAnfExpr("__builtin_Div__")

    def ParseFloorDiv(self, tree):
        return AtomicAnfExpr("__builtin_FloorDiv__")

    def ParseMod(self, tree):
        return AtomicAnfExpr("__builtin_Mod__")

    def ParseUSub(self, tree):
        return AtomicAnfExpr("__builtin_Neg__")

    def ParseEq(self, tree):
        return AtomicAnfExpr("__builtin_EQ__")

    def ParseNotEq(self, tree):
        return AtomicAnfExpr("__builtin_NE__")

    def ParseGt(self, tree):
        return AtomicAnfExpr("__builtin_GT__")

    def ParseGtE(self, tree):
        return AtomicAnfExpr("__builtin_GE__")

    def ParseLt(self, tree):
        return AtomicAnfExpr("__builtin_LT__")

    def ParseLtE(self, tree):
        return AtomicAnfExpr("__builtin_LE__")

    def ParseModule(self, module: ast.Module):
        parse_result = AnfParseResult(
            bindings=[], body_atomic_anf_expr=AtomicAnfExpr(None)
        )
        if len(module.body) > 0:
            seq_no_counter = itertools.count()
            return_count_constraint = ReturnCounterConstraint(limits=0)
            parse_result = functools.reduce(
                operator.add,
                (
                    PyToAnfParser(seq_no_counter, return_count_constraint)(tree)
                    for tree in module.body
                ),
            )
        return parse_result.ConvertToAnfExpr()

    def ParseFunctionDef(self, function_def: ast.FunctionDef):
        if len(function_def.body) > 0:
            return_count_constraint = ReturnCounterConstraint(limits=1)
            return_stmt_idx = self.GetStmtSizeUntilReturn(function_def.body)
            parse_result = functools.reduce(
                operator.add,
                [
                    PyToAnfParser(self.seq_no_counter, return_count_constraint)(
                        tree
                    )
                    for tree in function_def.body[0:return_stmt_idx]
                    if not isinstance(tree, ast.Pass)
                ]
                + [
                    AnfParseResult(
                        bindings=[], body_atomic_anf_expr=AtomicAnfExpr(None)
                    )
                ],
            )
        else:
            parse_result = AnfParseResult(
                bindings=[], body_atomic_anf_expr=AtomicAnfExpr(None)
            )
        args = [arg.arg for arg in function_def.args.args]
        lmbd = AtomicAnfExpr(
            ['lambda', args, parse_result.ConvertToAnfExpr().value]
        )
        for elt in reversed(function_def.decorator_list):
            decorator = self.Parse(elt)
            lmbd = self.BindToTmpVar([decorator.value, lmbd.value])
        func_name = function_def.name
        self.Bind(func_name, lmbd)
        return AtomicAnfExpr(func_name)

    def ParseLambda(self, function_def: ast.Lambda):
        return_count_constraint = ReturnCounterConstraint(limits=0)
        parser = PyToAnfParser(self.seq_no_counter, return_count_constraint)
        parse_result = parser(function_def.body)
        args = [arg.arg for arg in function_def.args.args]
        return AtomicAnfExpr(
            ['lambda', args, parse_result.ConvertToAnfExpr().value]
        )

    def ParseIfExp(self, if_expr: ast.IfExp):
        test_value = self.Parse(if_expr.test)
        true_value = self.ParseExprTo0ArgLambda(if_expr.body)
        false_value = self.ParseExprTo0ArgLambda(if_expr.orelse)
        ret = self.BindToTmpVar(
            [
                '__builtin_if__',
                test_value.value,
                true_value.value,
                false_value.value,
            ]
        )
        return ret

    def ParseBoolOp(self, bool_op: ast.BoolOp):
        name = type(bool_op.op).__name__
        method = f"Parse{name}"
        return getattr(self, method)(bool_op)

    def ParseOr(self, bool_op: ast.BoolOp):
        assert len(bool_op.values) == 2
        test_value = self.Parse(bool_op.values[0])
        true_value = AtomicAnfExpr(['lambda', [], AtomicAnfExpr(True).value])
        false_value = self.ParseExprTo0ArgLambda(bool_op.values[1])
        ret = self.BindToTmpVar(
            [
                '__builtin_if__',
                test_value.value,
                true_value.value,
                false_value.value,
            ]
        )
        return ret

    def ParseAnd(self, bool_op: ast.BoolOp):
        assert len(bool_op.values) == 2
        test_value = self.Parse(bool_op.values[0])
        true_value = self.ParseExprTo0ArgLambda(bool_op.values[1])
        false_value = AtomicAnfExpr(['lambda', [], AtomicAnfExpr(False).value])
        ret = self.BindToTmpVar(
            [
                '__builtin_if__',
                test_value.value,
                true_value.value,
                false_value.value,
            ]
        )
        return ret

    def ParseNot(self, unary_op: ast.UnaryOp):
        return AtomicAnfExpr('__builtin_not__')

    def ParseExprTo0ArgLambda(self, expr):
        return_count_constraint = ReturnCounterConstraint(limits=0)
        parser = PyToAnfParser(self.seq_no_counter, return_count_constraint)
        parse_result = parser(expr)
        return AtomicAnfExpr(
            ['lambda', [], parse_result.ConvertToAnfExpr().value]
        )

    def ParseAssert(self, expr: ast.Assert):
        test_value = self.Parse(expr.test)
        true_value = AtomicAnfExpr(['lambda', [], AtomicAnfExpr(None).value])
        # handle lambda: rase(msg)
        return_count_constraint = ReturnCounterConstraint(limits=0)
        parser = PyToAnfParser(self.seq_no_counter, return_count_constraint)
        if expr.msg is None:
            msg = parser.BindToTmpVar(AtomicAnfExpr({"str": ""}))
        else:
            msg = parser.Parse(expr.msg)
        exception = parser.BindToTmpVar(['AssertionError', msg.value])
        raise_ret = parser.BindToTmpVar(['raise', exception.value])
        false_value = AtomicAnfExpr(
            [
                'lambda',
                [],
                AnfParseResult(
                    bindings=parser.bindings, body_atomic_anf_expr=raise_ret
                )
                .ConvertToAnfExpr()
                .value,
            ]
        )
        ret = self.BindToTmpVar(
            [
                '__builtin_if__',
                test_value.value,
                true_value.value,
                false_value.value,
            ]
        )
        return ret

    def ParseAssign(self, tree):
        assert len(tree.targets) == 1
        if isinstance(tree.targets[0], ast.Name):
            val = self.Parse(tree.value)
            var = tree.targets[0].id
            self.Bind(var, val)
            return AtomicAnfExpr(var)
        elif isinstance(tree.targets[0], ast.Attribute):
            val = self.Parse(tree.value)
            attr = tree.targets[0]
            f = self.BindToTmpVar(
                [
                    '__builtin_setattr__',
                    self.Parse(attr.value).value,
                    {"str": attr.attr},
                ]
            )
            return self.BindToTmpVar([f.value, {"str": attr.attr}, val.value])
        elif isinstance(tree.targets[0], ast.Subscript):
            val = self.Parse(tree.value)
            subscript = tree.targets[0]
            slice_val = self.Parse(subscript.slice).value
            f = self.BindToTmpVar(
                [
                    '__builtin_setitem__',
                    self.Parse(subscript.value).value,
                    slice_val,
                ]
            )
            return self.BindToTmpVar([f.value, slice_val, val.value])
        else:
            raise NotImplementedError(tree.targets)

    def ParseSubscript(self, tree):
        val = self.Parse(tree.value)
        slc = self.Parse(tree.slice)
        return self.BindToTmpVar(["__builtin_getitem__", val.value, slc.value])

    def ParseExpr(self, tree):
        return self.BindToTmpVar(self.Parse(tree.value))

    def BindToTmpVar(self, value):
        tmp_var = self.get_tmp_var()
        self.Bind(tmp_var, value)
        return AtomicAnfExpr(tmp_var)

    def GetStmtSizeUntilReturn(self, stmts):
        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Return):
                return idx + 1
        return len(stmts)

    def ParseReturn(self, tree: ast.Return):
        self.return_count_constraint.CountAndCheck()
        value = self.Parse(tree.value)
        return self.BindToTmpVar(["__builtin_return__", value.value])

    def ParseStarred(self, tree: ast.Starred):
        value = self.Parse(tree.value)
        return self.BindToTmpVar(["__builtin_starred__", value.value])

    def ParseCall(self, tree: ast.Call):
        func = self.Parse(tree.func)
        assert isinstance(func, AtomicAnfExpr)

        def ParseArg(arg):
            parsed_arg = self.Parse(arg)
            assert isinstance(parsed_arg, AtomicAnfExpr)
            return parsed_arg

        args = [ParseArg(arg).value for arg in tree.args]
        kwargs = None
        if len(tree.keywords) > 0:
            keywords = [ParseArg(arg).value for arg in tree.keywords]
            kwargs = self.BindToTmpVar(["__builtin_list__", *keywords])
        if kwargs is None:
            if any(isinstance(arg, ast.Starred) for arg in tree.args):
                l = self.BindToTmpVar(["__builtin_list__", *args])
                return self.BindToTmpVar(
                    ["__builtin_apply__", func.value, l.value]
                )
            else:
                return self.BindToTmpVar([func.value, *args])
        else:
            args = self.BindToTmpVar(["__builtin_list__", *args])
            packed_args = self.BindToTmpVar(
                ["__builtin_PackedArgs__", args.value, kwargs.value]
            )
            return self.BindToTmpVar([func.value, packed_args.value])

    def ParseList(self, lst: ast.List):
        return self._ParseCall('__builtin_list__', lst.elts)

    def _ParseCall(self, func, ast_args):
        def ParseArg(arg):
            parsed_arg = self.Parse(arg)
            assert isinstance(parsed_arg, AtomicAnfExpr)
            return parsed_arg

        args = [ParseArg(arg).value for arg in ast_args]
        ret_var = self.get_tmp_var()
        self.Bind(ret_var, [func, *args])
        return AtomicAnfExpr(ret_var)

    def ParseAttribute(self, attr: ast.Attribute):
        ret_var = self.get_tmp_var()
        self.Bind(
            ret_var,
            [
                '__builtin_getattr__',
                self.Parse(attr.value).value,
                {"str": attr.attr},
            ],
        )
        return AtomicAnfExpr(ret_var)

    def ParseName(self, name: ast.Name):
        return AtomicAnfExpr(name.id)

    def ParseConstant(self, constant: ast.Constant):
        if isinstance(constant.value, str):
            return AtomicAnfExpr({"str": constant.value})
        if isinstance(constant.value, (bool, int, float)):
            return AtomicAnfExpr(constant.value)
        if constant.value is None:
            return AtomicAnfExpr(None)
        raise NotImplementedError(f"{constant} not supported by anf_expr")

    def ParseJoinedStr(self, tree: ast.JoinedStr):
        if len(tree.values) == 0:
            return AtomicAnfExpr({"str": ""})

        def ToString(elt):
            parsed_elt = self.Parse(elt)
            parsed_elt = self.BindToTmpVar(
                ['__builtin_ToString__', parsed_elt.value]
            )
            return parsed_elt

        ret = ToString(tree.values[0])
        for elt in tree.values[1:]:
            parsed_elt = ToString(elt)
            ret = self.BindToTmpVar(
                ['__builtin_Add__', ret.value, parsed_elt.value]
            )
        return ret

    def ParseFormattedValue(self, tree: ast.FormattedValue):
        return self.Parse(tree.value)

    def Bind(self, var_name, anf_expr):
        return getattr(self, f"Bind{type(anf_expr).__name__}")(
            var_name, anf_expr
        )

    def BindAtomicAnfExpr(self, var_name, anf_expr):
        self.bindings.append(
            [var_name, ["__builtin_identity__", anf_expr.value]]
        )

    def Bindlist(self, var_name, anf_expr):
        self.bindings.append([var_name, anf_expr])

    def get_tmp_var(self):
        return f"___{next(self.seq_no_counter)}"


class ReturnCounterConstraint:
    def __init__(self, limits):
        self.counter = itertools.count()
        self.limits = limits

    def CountAndCheck(self):
        return_stmt_id = next(self.counter)
        assert return_stmt_id < self.limits
