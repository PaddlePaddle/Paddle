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

import typing as t
from collections import OrderedDict

from paddle.incubate.cc.py_code_gen.ir import ir_constraint


class ConstraintTrait:

    def add_eq_cstr(self, lhs, rhs):
        if lhs == rhs:
            return
        cstr = ir_constraint.EqualConstraintRecord(lhs, rhs)
        self.get_eq_cstrs().append(cstr)

    def add_broadcastable_cstr(self, lhs, rhs):
        if lhs == rhs:
            return
        cstr = ir_constraint.BroadcastableConstraintRecord(lhs, rhs)
        self.get_broadcastable_cstrs().append(cstr)

    def add_gt_one_cstr(self, dim_expr):
        cstr = ir_constraint.GtOneConstraintRecord(dim_expr)
        self.get_gt_one_cstrs().append(cstr)

    def get_eq_cstrs(self):
        if not hasattr(self, "eq_cstrs_"):
            self.eq_cstrs_ = []
        return self.eq_cstrs_

    def get_broadcastable_cstrs(self):
        if not hasattr(self, "broadcastable_cstrs_"):
            self.broadcastable_cstrs_ = []
        return self.broadcastable_cstrs_

    def get_gt_one_cstrs(self):
        if not hasattr(self, "gt_one_cstrs_"):
            self.gt_one_cstrs_ = []
        return self.gt_one_cstrs_

    def clear_all_cstrs(self):
        del self.eq_cstrs_
        del self.broadcastable_cstrs_
        del self.gt_one_cstrs_

    def CollectConstraints(self):
        def MakePairs(cstrs):
            return [(cstr.lhs, cstr.rhs) for cstr in cstrs]

        return (
            [
                ir_constraint.EqualConstraint(dim_exprs)
                for dim_exprs in _MakeFindSets(MakePairs(self.get_eq_cstrs()))
            ]
            + [
                ir_constraint.BroadcastableConstraint(dim_exprs)
                for dim_exprs in _MakeFindSets(
                    MakePairs(self.get_broadcastable_cstrs())
                )
            ]
            + [
                ir_constraint.GtOneConstraint(dim_expr)
                for dim_expr in self.get_gt_one_cstrs()
            ]
        )


def _MakeFindSets(cstrs: list[tuple[t.Any, t.Any]]) -> list[list[t.Any]]:
    node2parent = {}
    for lhs, rhs in cstrs:
        node2parent[rhs] = lhs

    def FindRoot(x):
        if x not in node2parent:
            return x
        return FindRoot(node2parent[x])

    root2clusters = OrderedDict()

    def GetCluster(root):
        if root not in root2clusters:
            root2clusters[root] = OrderedDict({root: True})
        return root2clusters[root]

    for lhs, rhs in cstrs:
        GetCluster(FindRoot(lhs))[lhs] = True
        GetCluster(FindRoot(rhs))[rhs] = True
    return [list(cluster) for root, cluster in root2clusters.items()]
