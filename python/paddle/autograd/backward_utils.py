# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,tes
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import warnings

from paddle.base import core
from paddle.base.wrapped_decorator import signature_safe_contextmanager


class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be bining with a program

    """

    def __init__(self, block):
        self.block = block
        # opresult -> list(list(opresult))
        self.value_to_valuegrad = collections.defaultdict(list)
        self.value_to_sumvaluegrad = collections.defaultdict(list)
        # operation -> list(operation)
        self.op_to_opgrad = collections.defaultdict(list)

        # opresult -> list(opresult)
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        # operation -> list(operation)
        self.opgrad_to_op = collections.defaultdict(list)

    def turn_map(self) -> None:
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        self.opgrad_to_op = collections.defaultdict(list)

        for k, v in self.value_to_valuegrad.items():
            if v != []:
                for value in v[0]:
                    self.valuegrad_to_value[value] = [k]
        for k, v in self.value_to_sumvaluegrad.items():
            if v != []:
                for value in v[0]:
                    self.sumvaluegrad_to_value[value] = [k]
        for k, v in self.op_to_opgrad.items():
            if v != []:
                self.opgrad_to_op[v[0]] = [k]

    def copy(self, new_block):
        state = State(new_block)
        state.value_to_valuegrad = self.value_to_valuegrad.copy()
        state.value_to_sumvaluegrad = self.value_to_sumvaluegrad.copy()

        # operation -> list(operation)
        state.op_to_opgrad = self.op_to_opgrad.copy()

        # opresult -> list(opresult)
        state.valuegrad_to_value = self.valuegrad_to_value.copy()
        state.sumvaluegrad_to_value = self.sumvaluegrad_to_value.copy()
        # operation -> list(operation)
        state.opgrad_to_op = self.opgrad_to_op.copy()

        return state


def _check_vjp_dynamic_shape(op, inputs):
    for items in inputs:
        for item in items:
            shape = item.shape
            if -1 in shape:
                warnings.warn(
                    f"[Prim] Decomp op does not support dynamic shape -1, but got shape {item.shape} in inputs of op {op.name()} . Prim will skip its vjp op."
                )
                return True


# Prim currently does not support dynamic shape, when dynamic shape exits in shape of op inputs, prim will be skipped its vjp op.
@signature_safe_contextmanager
def dynamic_shape_prim_vjp_guard(op, inputs):
    skip_prim = (
        core._is_bwd_prim_enabled()
        and core._enable_prim_dynamic_shape()
        and _check_vjp_dynamic_shape(op, inputs)
    )
    try:
        if skip_prim:
            core._set_prim_backward_enabled(False)
        yield
    finally:
        if skip_prim:
            core._set_prim_backward_enabled(True)
