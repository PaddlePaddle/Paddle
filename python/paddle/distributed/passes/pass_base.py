# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import sys
from abc import ABC, abstractmethod
from paddle.fluid.framework import program_guard, _apply_pass as _apply_cpp_pass


class PassContext:
    def __init__(self):
        self._applied_passes = []
        self._attrs = []

    def set_attr(self, key, value):
        self._attrs[key] = value

    def get_attr(self, key, default=None):
        return self._attrs.get(key, default)

    @property
    def passes(self):
        return self._applied_passes

    def _add_pass(self, pass_obj):
        self._applied_passes.append(pass_obj)

    def _pop_pass(self):
        del self._applied_passes[-1]


class PassType:
    UNKNOWN = 0
    COMM_OPT = 1
    CALC_OPT = 2
    PARALLEL_OPT = 3
    FUSION_OPT = 4


class PassBase(ABC):
    _REGISTERED_PASSES = {}
    _COMMON_RULES = []
    # TODO(zengjinle): add white/black list

    name = None

    @staticmethod
    def _register(pass_name, pass_class):
        assert issubclass(pass_class, PassBase)
        PassBase._REGISTERED_PASSES[pass_name] = pass_class

    def __init__(self):
        self._attrs = {}

    def set_attr(self, key, value):
        self._attrs[key] = value
        return self

    def get_attr(self, key, default=None):
        return self._attrs.get(key, default)

    @abstractmethod
    def _check_self(self):
        pass

    @abstractmethod
    def _check_conflict(self, other_pass):
        pass

    def _type(self):
        return PassType.UNKNOWN

    def _check_conflict_including_common_rules(self, other_pass):
        return self._check_conflict(other_pass) and all(
            [r(other_pass, self) for r in PassBase._COMMON_RULES])

    def apply(self, main_programs, startup_programs, context=None):
        if context is None:
            context = PassContext()

        if not self._check_self():
            return context

        if not all([
                self._check_conflict_including_common_rules(p)
                for p in context.passes
        ]):
            return context

        assert isinstance(main_programs, list)
        assert isinstance(startup_programs, list)
        assert len(main_programs) == len(startup_programs)
        self._apply_impl(main_programs, startup_programs, context)
        context._add_pass(self)
        return context

    def _apply_impl(self, main_programs, startup_programs, context):
        for main_program, startup_program in zip(main_programs,
                                                 startup_programs):
            self._apply_single_impl(main_program, startup_program, context)

    @abstractmethod
    def _apply_single_impl(self, main_program, startup_program, context):
        pass


def register_pass(name):
    def impl(cls):
        PassBase._register(name, cls)
        cls.name = name
        return cls

    return impl


def new_pass(name, pass_attrs={}):
    pass_class = PassBase._REGISTERED_PASSES.get(name)
    assert pass_class is not None, "Pass {} is not registered".format(name)
    pass_obj = pass_class()
    for k, v in pass_attrs.items():
        pass_obj.set_attr(k, v)
    return pass_obj


class CPPPassWrapper(PassBase):
    def __init__(self):
        super(CPPPassWrapper, self).__init__()

    @property
    def cpp_name(self):
        raise NotImplementedError()

    @property
    def cpp_attr_types(self):
        return {}

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        _apply_cpp_pass(main_program, startup_program, self.cpp_name,
                        self._attrs, self.cpp_attr_types)


def _fusion_opt_last_rule(pass_before, pass_after):
    if pass_before._type() == PassType.FUSION_OPT and pass_after._type(
    ) != PassType.FUSION_OPT:
        return False
    else:
        return True


PassBase._COMMON_RULES = [
    _fusion_opt_last_rule,
    lambda pass_before, pass_after: type(pass_before) != type(pass_after),
    # Add more common rules here
]


def _find_longest_path(edges):
    n = len(edges)
    paths = [None] * n
    dists = [None] * n

    min_path = []
    min_dist = 0
    for i in range(n):
        paths[i] = [None] * n
        dists[i] = [None] * n
        for j in range(n):
            assert isinstance(edges[i][j], bool)
            if not edges[i][j]:
                dists[i][j] = n  # inf
                paths[i][j] = []
            else:
                assert edges[i][j] is True
                dists[i][j] = -1
                paths[i][j] = [i, j]
                if dists[i][j] < min_dist:
                    min_dist = -1
                    min_path = paths[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dists[i][j] > dists[i][k] + dists[k][j]:
                    dists[i][j] = dists[i][k] + dists[k][j]
                    paths[i][j] = paths[i][k] + paths[k][j]
                    if dists[i][j] < min_dist:
                        min_dist = dists[i][j]
                        min_path = paths[i][j]

    return min_path if min_path else [0]


def _solve_pass_conflict(passes, context):
    passes = [p for p in passes if p._check_self()]
    if not passes:
        return []

    old_passes = passes
    passes = []
    for p in old_passes:
        if all([
                p._check_conflict_including_common_rules(applied_p)
                for applied_p in context.passes
        ]):
            passes.append(p)

    if not passes:
        return []

    n = len(passes)
    adjacent_matrix = []
    for _ in range(n):
        adjacent_matrix.append([None] * n)

    for i in range(n):
        for j in range(n):
            adjacent_matrix[i][j] = passes[
                j]._check_conflict_including_common_rules(passes[i])

    longest_path = _find_longest_path(adjacent_matrix)
    return [passes[idx] for idx in longest_path]


class PassManager:
    def __init__(self, passes, context=None, auto_solve_conflict=True):
        if context is None:
            context = PassContext()
        self._context = context

        if auto_solve_conflict:
            self._passes = _solve_pass_conflict(passes, context)
        else:
            self._passes = list(passes)

    def apply(self, main_programs, startup_programs):
        context = self._context
        for p in self._passes:
            context = p.apply(main_programs, startup_programs, context)
        self._context = context
        return context

    @property
    def context(self):
        return self._context

    @property
    def names(self):
        return [p.name for p in self._passes]
