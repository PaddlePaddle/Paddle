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
from dataclasses import dataclass


@dataclass
class OpCall:
    op: Op  # noqa: F821
    input_tensors: list[Tensor]  # noqa: F821
    kwargs: dict[str, t.Any]


@dataclass
class BlockOpCalls:
    input_op_calls: list[OpCall]
    body_op_calls: list[OpCall]
    output_op_calls: list[OpCall]


class BlockOpCallsExtractor:

    def __init__(self):
        self.block_op_calls = BlockOpCalls([], [], [])

    def Extract(self, func, free_vars, args) -> BlockOpCalls:
        func(self, *free_vars)(*args)
        return self.block_op_calls

    def cf_yield(self, op, *inputs):
        self.block_op_calls.output_op_calls.append(
            OpCall(op=op, input_tensors=inputs, kwargs={})
        )
        return op.GetResults()

    def builtin_shadow_output(self, op, *inputs):
        self.block_op_calls.output_op_calls.append(
            OpCall(op=op, input_tensors=inputs, kwargs={})
        )
        return op.GetResults()

    def pd_op_fetch(self, op, *inputs):
        self.block_op_calls.output_op_calls.append(
            OpCall(op=op, input_tensors=inputs, kwargs={})
        )
        return op.GetResults()

    def builtin_parameter(self, op):
        self.block_op_calls.input_op_calls.append(
            OpCall(op=op, input_tensors=[], kwargs={})
        )
        return op.GetResults()

    def builtin_constant(self, op):
        self.block_op_calls.input_op_calls.append(
            OpCall(op=op, input_tensors=[], kwargs={})
        )
        return op.GetResults()

    def pd_op_data(self, op):
        self.block_op_calls.input_op_calls.append(
            OpCall(op=op, input_tensors=[], kwargs={})
        )
        return op.GetResults()

    def pd_op_feed(self, op):
        self.block_op_calls.input_op_calls.append(
            OpCall(op=op, input_tensors=[], kwargs={})
        )
        return op.GetResults()

    def __call__(self, op, *input_tensors, **kwargs):
        if hasattr(self, op.GetPyVarName()):
            return getattr(self, op.GetPyVarName())(
                op, *input_tensors, **kwargs
            )
        self.block_op_calls.body_op_calls.append(
            OpCall(
                op=op,
                input_tensors=input_tensors,
                kwargs=kwargs,
            )
        )
        return op.GetResults()
