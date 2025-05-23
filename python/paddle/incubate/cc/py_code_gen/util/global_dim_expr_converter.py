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

from collections import OrderedDict
from dataclasses import dataclass

import networkx as nx

from paddle.incubate.cc.py_code_gen.ir import ir_symbol
from paddle.incubate.cc.py_code_gen.util.hash_combine import hash_combine
from paddle.incubate.cc.py_code_gen.util.new_symbol_generator import (
    NewSymbolGenerator,
)


class GlobalDimExprConverter:

    def __init__(self, dim_exprs, new_symbol_generator=None):
        if new_symbol_generator is None:
            new_symbol_generator = NewSymbolGenerator()
        global_dim_expr2local_dim_expr = OrderedDict(
            (dim_expr, dim_expr) for dim_expr in dim_exprs
        )
        single_dim_expr_simplifier = SingleDimExprSimplifier(
            new_symbol_generator
        )
        global_dim_expr2local_dim_expr = single_dim_expr_simplifier.Simplify(
            global_dim_expr2local_dim_expr
        )
        multi_dim_expr_simplifier = MultiDimExprSimplifier(new_symbol_generator)
        global_dim_expr2local_dim_expr = multi_dim_expr_simplifier.Simplify(
            global_dim_expr2local_dim_expr
        )
        self.global_dim_expr2local_dim_expr = global_dim_expr2local_dim_expr

    def GetLocalDimExpr(self, dim_expr):
        if dim_expr not in self.global_dim_expr2local_dim_expr:
            return dim_expr
        return self.global_dim_expr2local_dim_expr[dim_expr]


class MultiDimExprSimplifier:

    def __init__(self, new_symbol_generator=None):
        if new_symbol_generator is None:
            new_symbol_generator = NewSymbolGenerator()
        self.new_symbol_generator = new_symbol_generator

    def Simplify(self, key2dim_expr):
        key2dim_expr = dict(key2dim_expr)
        while True:
            dim_exprs, container_dim_exprs = (
                self.GetOneDimExprTupleCanbeSubstituted(key2dim_expr)
            )
            if dim_exprs == ():
                break
            substitutor = MultiDimExprSubstitutor(
                pattern_dim_exprs=dim_exprs,
                container_dim_exprs=container_dim_exprs,
                replaced_dim_expr=self.new_symbol_generator.Mangle(
                    self.GetSoleDimExprCls(container_dim_exprs)(dim_exprs)
                ),
            )
            substitutor.SubstituteDictValueDimExprs(key2dim_expr)
        return key2dim_expr

    def GetSoleDimExprCls(self, dim_exprs):
        cls = {type(x) for x in dim_exprs}
        assert len(cls) == 1
        return next(iter(cls))

    def GetOneDimExprTupleCanbeSubstituted(self, key2dim_expr):

        # This function must handle corner cases like {a:S0*S1, b:S0*S1, c:S0}
        # whose return value should be ((), ()) rather than ((S0, S1), S0*S1)
        # Make sure that every ir_symbol.String has at least one container by adding zero.
        dim_exprs = [
            ir_symbol.Add([ir_symbol.Int64(0), dim_expr])
            for _, dim_expr in key2dim_expr.items()
        ]

        dim_expr2container_dim_exprs = OrderedDict()
        for dim_expr in dim_exprs:
            self.CollectDimExpr2ContainerDimExprs(
                dim_expr, dim_expr2container_dim_exprs
            )
        container_dim_exprs2dim_exprs = OrderedDict()
        for (
            dim_expr,
            container_dim_exprs,
        ) in dim_expr2container_dim_exprs.items():
            if container_dim_exprs not in container_dim_exprs2dim_exprs:
                container_dim_exprs2dim_exprs[container_dim_exprs] = []
            container_dim_exprs2dim_exprs[container_dim_exprs].append(dim_expr)
        for (
            container_dim_exprs,
            dim_exprs,
        ) in container_dim_exprs2dim_exprs.items():
            # ir_symbol.String only
            dim_exprs = tuple(
                x for x in dim_exprs if isinstance(x, ir_symbol.String)
            )
            if len(dim_exprs) <= 1:
                continue
            # same type of container
            if len({type(x) for x in container_dim_exprs}) > 1:
                continue
            filtered_dim_exprs = {
                self.FilterOperands(container_dim_expr, dim_exprs)
                for container_dim_expr in container_dim_exprs
            }
            if len(filtered_dim_exprs) > 1:
                continue
            return next(iter(filtered_dim_exprs)), container_dim_exprs
        return (), ()

    def CollectDimExpr2ContainerDimExprs(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        method_name = (
            f"CollectDimExpr2ContainerDimExprs_{type(dim_expr).__name__}"
        )
        return getattr(self, method_name)(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Int64(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        pass

    def CollectDimExpr2ContainerDimExprs_String(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        pass

    def CollectDimExpr2ContainerDimExprs_Negative(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2UnaryContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Reciprocal(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2UnaryContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Add(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2VariadicContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Mul(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2VariadicContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Max(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2VariadicContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Min(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2VariadicContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2ContainerDimExprs_Broadcast(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2VariadicContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2UnaryContainerDimExprs(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        self.CollectDimExpr2UnaryContainerDimExpr(
            dim_expr.operand, dim_expr, dim_expr2container_dim_exprs
        )

    def CollectDimExpr2VariadicContainerDimExprs(
        self, dim_expr, dim_expr2container_dim_exprs
    ):
        for operand in dim_expr.operands:
            self.CollectDimExpr2UnaryContainerDimExpr(
                operand, dim_expr, dim_expr2container_dim_exprs
            )

    def CollectDimExpr2UnaryContainerDimExpr(
        self, dim_expr, container_dim_expr, dim_expr2container_dim_exprs
    ):
        if dim_expr not in dim_expr2container_dim_exprs:
            dim_expr2container_dim_exprs[dim_expr] = ()
        if container_dim_expr not in dim_expr2container_dim_exprs[dim_expr]:
            dim_expr2container_dim_exprs[dim_expr] += (container_dim_expr,)
        self.CollectDimExpr2ContainerDimExprs(
            dim_expr, dim_expr2container_dim_exprs
        )

    # e.g.1
    #   container_dim_expr: S0 * S0 * 30
    #   dim_exprs: S0
    #   return: (S0, S0)
    def FilterOperands(self, container_dim_expr, dim_exprs):
        return tuple(
            operand
            for operand in container_dim_expr.operands
            if operand in dim_exprs
        )


class MultiDimExprSubstitutor:
    def __init__(
        self, pattern_dim_exprs, container_dim_exprs, replaced_dim_expr
    ):
        self.pattern_dim_exprs = pattern_dim_exprs
        self.container_dim_exprs = container_dim_exprs
        self.replaced_dim_expr = replaced_dim_expr

    def SubstituteDictValueDimExprs(self, key2dim_expr):
        key2substituted = OrderedDict()
        for key, dim_expr in key2dim_expr.items():
            substituted = self.Substitute(dim_expr)
            if substituted is not None:
                key2substituted[key] = substituted
        key2dim_expr.update(key2substituted)

    def Substitute(self, dim_expr):
        method_name = f"Substitute_{type(dim_expr).__name__}"
        return getattr(self, method_name)(dim_expr)

    def Substitute_Int64(self, dim_expr):
        return None

    def Substitute_String(self, dim_expr):
        return None

    def Substitute_Negative(self, dim_expr):
        return self.SubstituteUnary(dim_expr, ir_symbol.Negative)

    def Substitute_Reciprocal(self, dim_expr):
        return self.SubstituteUnary(dim_expr, ir_symbol.Reciprocal)

    def SubstituteUnary(self, dim_expr, cls):
        substituted_operand = self.Substitute(dim_expr.operand)
        if substituted_operand is None:
            return None
        return cls(substituted_operand)

    def Substitute_Add(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Add)

    def Substitute_Mul(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Mul)

    def Substitute_Max(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Max)

    def Substitute_Min(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Min)

    def Substitute_Broadcast(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Broadcast)

    def SubstituteVariadic(self, dim_expr, cls):
        if dim_expr not in self.container_dim_exprs:
            substituted_once = False
            substituted_operands = []
            for operand in dim_expr.operands:
                substituted_operand = self.Substitute(operand)
                substituted_once = substituted_once or (
                    substituted_operand is not None
                )
                substituted_operands.append(
                    substituted_operand
                    if substituted_operand is not None
                    else operand
                )
            return cls(substituted_operands)
        else:
            substituted_operands = []
            for operand in dim_expr.operands:
                if operand in self.pattern_dim_exprs:
                    continue
                substituted_operand = self.Substitute(operand)
                substituted_operands.append(
                    substituted_operand
                    if substituted_operand is not None
                    else operand
                )
            if len(substituted_operands) == 0:
                return self.replaced_dim_expr
            elif len(substituted_operands) == 1:
                return cls([self.replaced_dim_expr, substituted_operands[0]])
            else:
                return cls([self.replaced_dim_expr, cls(substituted_operands)])


class SingleDimExprSimplifier:

    def __init__(self, new_symbol_generator=None):
        if new_symbol_generator is None:
            new_symbol_generator = NewSymbolGenerator()
        self.new_symbol_generator = new_symbol_generator

    def Simplify(self, key2dim_expr):
        key2dim_expr = dict(key2dim_expr)
        while True:
            dim_expr = self.GetOneDimExprCanbeSubstituted(key2dim_expr)
            if dim_expr is None:
                break
            substitutor = SingleDimExprSubstitutor(
                pattern_dim_expr=dim_expr,
                replaced_dim_expr=self.new_symbol_generator.Mangle(dim_expr),
            )
            substitutor.SubstituteDictValueDimExprs(key2dim_expr)

        return key2dim_expr

    def GetOneDimExprCanbeSubstituted(self, key2dim_expr):
        dim_expr_containing_graph = self.MakeDimExprContainingGraph(
            key2dim_expr
        )
        for _, cut_node in self.GetDimExprUsingBridges(
            dim_expr_containing_graph
        ):
            dim_expr = self.GetNodeDimExpr(cut_node)
            if dim_expr.IsConstant():
                continue
            if isinstance(dim_expr, ir_symbol.String):
                continue
            return dim_expr
        return None

    def GetDimExprUsingBridges(self, dim_expr_containing_graph):
        for src, dst in nx.bridges(dim_expr_containing_graph):
            if isinstance(dst, DimExprValueNode):
                yield src, dst

    def GetNodeDimExpr(self, node):
        return node.dim_expr

    def MakeDimExprContainingGraph(self, key2dim_expr):
        graph = nx.Graph()
        root = RootNode()
        for _, dim_expr in key2dim_expr.items():
            self.RecursivelyAddEdges(graph, root, dim_expr)
        return graph

    def AddAllContainingEdges(self, graph, node):
        method_name = f"AddEdge_{type(node.dim_expr).__name__}"
        return getattr(self, method_name)(graph, node)

    def AddEdge_Int64(self, graph, node):
        pass

    def AddEdge_String(self, graph, node):
        pass

    def AddEdge_Negative(self, graph, node):
        self.AddEdgeForUnary(graph, node)

    def AddEdge_Reciprocal(self, graph, node):
        self.AddEdgeForUnary(graph, node)

    def AddEdge_Add(self, graph, node):
        self.AddEdgeForVariadic(graph, node)

    def AddEdge_Mul(self, graph, node):
        self.AddEdgeForVariadic(graph, node)

    def AddEdge_Max(self, graph, node):
        self.AddEdgeForVariadic(graph, node)

    def AddEdge_Min(self, graph, node):
        self.AddEdgeForVariadic(graph, node)

    def AddEdge_Broadcast(self, graph, node):
        self.AddEdgeForVariadic(graph, node)

    def AddEdgeForUnary(self, graph, node):
        self.RecursivelyAddEdges(graph, node, node.dim_expr.operand)

    def AddEdgeForVariadic(self, graph, node):
        for operand in node.dim_expr.operands:
            self.RecursivelyAddEdges(graph, node, operand)

    def RecursivelyAddEdges(self, graph, node, dim_expr):
        if dim_expr.IsConstant():
            return
        value_node = self.AddEdgeTo(graph, node, dim_expr)
        self.AddAllContainingEdges(graph, value_node)

    def AddEdgeTo(self, graph, node, dim_expr):
        var_node = DimExprVarNode(dim_expr)
        value_node = DimExprValueNode(dim_expr)
        graph.add_edge(node, var_node)
        graph.add_edge(var_node, value_node)
        return value_node


@dataclass
class Node:
    pass


@dataclass
class RootNode(Node):
    pass

    def __hash__(self):
        return id(RootNode)


@dataclass
class DimExprVarNode(Node):
    dim_expr: ir_symbol.DimExpr

    def __hash__(self):
        return hash_combine(id(DimExprVarNode), hash(self.dim_expr))


@dataclass
class DimExprValueNode(Node):
    dim_expr: ir_symbol.DimExpr

    def __hash__(self):
        return hash_combine(id(DimExprValueNode), hash(self.dim_expr))


class SingleDimExprSubstitutor:
    def __init__(self, pattern_dim_expr, replaced_dim_expr):
        self.pattern_dim_expr = pattern_dim_expr
        self.replaced_dim_expr = replaced_dim_expr

    def SubstituteDictValueDimExprs(self, key2dim_expr):
        key2substituted = OrderedDict()
        for key, dim_expr in key2dim_expr.items():
            substituted = self.Substitute(dim_expr)
            if substituted is not None:
                key2substituted[key] = substituted
        key2dim_expr.update(key2substituted)

    def Substitute(self, dim_expr):
        if dim_expr == self.pattern_dim_expr:
            return self.replaced_dim_expr
        method_name = f"Substitute_{type(dim_expr).__name__}"
        return getattr(self, method_name)(dim_expr)

    def Substitute_Int64(self, dim_expr):
        return None

    def Substitute_String(self, dim_expr):
        return None

    def Substitute_Negative(self, dim_expr):
        return self.SubstituteUnary(dim_expr, ir_symbol.Negative)

    def Substitute_Reciprocal(self, dim_expr):
        return self.SubstituteUnary(dim_expr, ir_symbol.Reciprocal)

    def SubstituteUnary(self, dim_expr, cls):
        substituted_operand = self.Substitute(dim_expr.operand)
        if substituted_operand is None:
            return None
        return cls(substituted_operand)

    def Substitute_Add(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Add)

    def Substitute_Mul(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Mul)

    def Substitute_Max(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Max)

    def Substitute_Min(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Min)

    def Substitute_Broadcast(self, dim_expr):
        return self.SubstituteVariadic(dim_expr, ir_symbol.Broadcast)

    def SubstituteVariadic(self, dim_expr, cls):
        substituted_once = False
        substituted_operands = []
        for operand in dim_expr.operands:
            substituted_operand = self.Substitute(operand)
            substituted_once = substituted_once or (
                substituted_operand is not None
            )
            substituted_operands.append(
                substituted_operand
                if substituted_operand is not None
                else operand
            )
        return cls(substituted_operands)
