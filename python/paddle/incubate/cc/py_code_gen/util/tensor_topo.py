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

import itertools
from collections import OrderedDict
from dataclasses import dataclass

from paddle.incubate.cc.py_code_gen.util.block_op_calls_extractor import (
    BlockOpCallsExtractor,
)
from paddle.incubate.cc.py_code_gen.util.input_output_tensors_extractor import (
    InputOutputTensorsExtractor,
)


@dataclass
class OpInpOutNameSignature:
    op_id: int
    in_names: list[str]
    out_names: list[str]


@dataclass
class OpPipeInOutNamesSignature:
    op_id: int
    in_names: list[str]
    out_names: list[str]


def GetOpId2OpPipeInOutNamesSignature(
    op_id2used_by_me_and_downstream,
    func,
    free_vars,
    args,
    get_local_name,
) -> dict[int, OpPipeInOutNamesSignature]:
    op_id2used = op_id2used_by_me_and_downstream
    if len(op_id2used) == 0:
        return {}
    extractor = InputOutputTensorsExtractor(func)
    input_tensors, output_tensors = extractor.Extract(free_vars, args)
    input_tensor_names = [get_local_name(t) for t in input_tensors]

    def get_in_names_list():
        in_names_list = [
            names_used_by_me_and_downstream
            for _, names_used_by_me_and_downstream in op_id2used.items()
        ]
        if len(in_names_list) > 0:
            in_names_list[0] = input_tensor_names
        return in_names_list

    def get_out_names_list():
        out_names_list = [
            names_used_by_me_and_downstream
            for _, names_used_by_me_and_downstream in op_id2used.items()
        ]
        out_names_list = out_names_list[1:]
        out_names_list += [get_local_name(t) for t in output_tensors]
        return out_names_list

    def get_op_id_list():
        return [op_id for op_id, _ in op_id2used.items()]

    op_id2op_pipe_in_out_names_sig = OrderedDict()
    tuple_list = zip(
        get_op_id_list(), get_in_names_list(), get_out_names_list()
    )
    for op_id, in_names, out_names in tuple_list:
        op_id2op_pipe_in_out_names_sig[op_id] = OpPipeInOutNamesSignature(
            op_id=op_id,
            in_names=in_names,
            out_names=out_names,
        )
    return op_id2op_pipe_in_out_names_sig


def GetOpId2TensorNamesUsedByMeAndDownstream(
    func,
    free_vars,
    args,
    get_local_name,
) -> dict[int, list[str]]:
    in_out_name_sig_extractor = OpInOutNameSignatureExtractor(get_local_name)
    in_out_names_sigs = in_out_name_sig_extractor.Extract(func, free_vars, args)
    input_tensors, output_tensors = InputOutputTensorsExtractor(func).Extract(
        free_vars, args
    )
    input_tensor_names = [get_local_name(tensor) for tensor in input_tensors]
    output_tensor_names = [get_local_name(tensor) for tensor in output_tensors]
    tensor_name2producer_idx = _GetTensorName2ProducerIdx(
        in_out_names_sigs,
        input_tensors,
        get_local_name,
    )
    op_id2used = OrderedDict()
    block_op_calls = BlockOpCallsExtractor().Extract(func, free_vars, args)
    for op_call in block_op_calls.input_op_calls:
        op_id2used[op_call.op.op_id] = input_tensor_names
    body_op_id2used_tensor_names = (
        GetOpId2DefinedTensorNamesUsedByMeAndDownstreams(
            in_out_names_sigs, output_tensor_names
        )
    )
    assert len(block_op_calls.body_op_calls) == len(in_out_names_sigs)
    counter = itertools.count()
    seq = itertools.count()
    for i, in_out_names_sig in enumerate(in_out_names_sigs):
        op_id = in_out_names_sig.op_id
        op_id2used[op_id] = body_op_id2used_tensor_names[op_id]
        op_id2used[op_id].sort(key=lambda name: tensor_name2producer_idx[name])
    for op_call in block_op_calls.output_op_calls:
        op_id2used[op_call.op.op_id] = output_tensor_names
    return op_id2used


def _GetSorted(names, key):
    return sorted(names, key=key)


def GetOpId2DefinedTensorNamesUsedByMeAndDownstreams(
    in_out_names_sigs,
    output_tensor_names,
):
    op_id2used_by_me_and_downstreams = {}
    used_by_downstream = set(output_tensor_names)
    for in_out_names_sig in reversed(in_out_names_sigs):
        used_by_downstream.difference_update(in_out_names_sig.out_names)
        used_by_downstream.update(in_out_names_sig.in_names)
        op_id2used_by_me_and_downstreams[in_out_names_sig.op_id] = list(
            used_by_downstream
        )
    return op_id2used_by_me_and_downstreams


def _GetTensorName2ProducerIdx(
    in_out_names_sigs, input_tensors, get_local_name
):
    tensor_name2idx = {}
    for j, input_tensor in enumerate(input_tensors):
        tensor_name2idx[get_local_name(input_tensor)] = (-1, j)
    for i, in_out_names_sig in enumerate(in_out_names_sigs):
        for j, out_name in enumerate(in_out_names_sig.out_names):
            if out_name in tensor_name2idx:
                continue
            tensor_name2idx[out_name] = (i, j)
    return tensor_name2idx


class OpInOutNameSignatureExtractor:

    def __init__(self, get_local_name):
        self.in_out_names_sigs = []
        self.get_local_name = get_local_name

    def Extract(self, func, free_vars, args):
        body_op_calls = (
            BlockOpCallsExtractor().Extract(func, free_vars, args).body_op_calls
        )
        for op_call in body_op_calls:
            self(op_call.op, op_call.input_tensors, op_call.kwargs)
        return self.in_out_names_sigs

    def __call__(self, op, input_tensors, kwargs):
        input_tensors = [t for t in input_tensors if t is not None]
        if hasattr(self, op.GetPyVarName()):
            return getattr(self, op.GetPyVarName())(
                op, *input_tensors, **kwargs
            )
        free_vars = []
        if len(kwargs) > 0:
            for region in kwargs["blocks"]:
                for block_tuple in region:
                    free_vars = [*free_vars, *block_tuple[1:]]
        input_tensors = [*free_vars, *input_tensors]
        in_names = [self.get_local_name(tensor) for tensor in input_tensors]
        out_names = [self.get_local_name(tensor) for tensor in op.GetResults()]
        self.in_out_names_sigs.append(
            OpInpOutNameSignature(
                op_id=op.op_id,
                in_names=in_names,
                out_names=out_names,
            )
        )
