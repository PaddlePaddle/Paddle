#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle

from ...fluid import core, framework, Program, program_guard, unique_name
from ...fluid.layers.utils import _hash_with_id
from ...fluid.framework import in_dygraph_mode
from ...common_ops_import import *

from collections.abc import Sequence, Mapping

__all__ = ["map", "data_reader"]


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return l
    return [l]


class _ProgramGuard(object):
    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("MapGuard should init with a Program")
        self._main_program = main_program

    def __enter__(self):
        self._main_program._create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._main_program._rollback()
        return exc_type is None


class _StreamIDGenerator(object):
    def __init__(self):
        self.stream_id = 0

    def get_stream_id(self):
        self.stream_id += 1
        return self.stream_id - 1


_stream_id_generator = _StreamIDGenerator()


def _generate_stream_id():
    return _stream_id_generator.get_stream_id()


def map(map_func, *args, **kwargs):
    if in_dygraph_mode():
        return map_func(inputs)

    helper = LayerHelper("map", **locals())

    # NOTE: map_func can take List(Tensor) (while batch_size > 1) as
    #       inputs or outputs, which means we need to keep the structure
    #       info when calling map_func, _build_program_inputs used to
    #       generate 3 kinds of infos:
    #       1. return value: holds variables in map_block, and keeps the
    #          structure info of map inputs, will be used to call map_func
    #       2. input_vars: holds variables in map_block in flatten format,
    #          will be used to generate input_var_names
    #       3. flat_inputs: holds variables in main_program/global_block in
    #          flatten format, will be used as inputs for appendding map OP
    #       and _parse_program_outputs follows similar logic
    def _build_program_inputs(inputs, map_block,
                              input_vars=[], flat_inputs=[]):
        if isinstance(inputs, Sequence):
            return [_build_program_inputs(inp, map_block, input_vars,
                        flat_inputs) for inp in inputs]
        elif isinstance(inputs, Mapping):
            return {k: _build_program_inputs(v, map_block, input_vars,
                        flat_inputs) for k,v in inputs.items()}
        else:
            var = map_block.create_var(
                        name=unique_name.generate("map_sub"),
                        type=inputs.desc.type(),
                        dtype=inputs.desc.dtype(),
                        persistable=False)
            input_vars.append(var)
            flat_inputs.append(inputs)
            return var

    def _parse_program_outputs(outputs, output_vars=[], flat_outputs=[]):
        if isinstance(outputs, Sequence):
            return [_parse_program_outputs(outp, output_vars,
                        flat_outputs) for outp in outputs]
        elif isinstance(outputs, Mapping):
            return {k: _parse_program_outputs(v, output_vars,
                        flat_outputs) for outp in outputs}
        else:
            var = helper.create_variable(
                            name=unique_name.generate("map"),
                            type=outputs.desc.type(),
                            dtype=outputs.desc.dtype(),
                            persistable=True)
            flat_outputs.append(var)
            output_vars.append(outputs)
            return var

    # build map block
    main_program = helper.main_program
    with _ProgramGuard(main_program):
        program_id = _hash_with_id(main_program, map_func)
        map_block = main_program.current_block()

        input_vars, flat_inputs = [], []
        program_inputs_args = _build_program_inputs(
                                args, map_block, input_vars, flat_inputs)
        program_inputs_kwargs = _build_program_inputs(
                                kwargs, map_block, input_vars, flat_inputs)

        program_outputs = map_func(*program_inputs_args,
                                   **program_inputs_kwargs)

    # NOTE: _parse_program_outputs create main_program variables, so
    #       we need to call it outside of _ProgramGuard
    output_vars, flat_outputs = [], []
    outputs = _parse_program_outputs(program_outputs, output_vars,
                                     flat_outputs)
    input_var_names = [v.name for v in input_vars]
    output_var_names = [v.name for v in output_vars]

    attrs = {
        "map_block": map_block,
        "program_id": program_id,
        "input_var_names": input_var_names,
        "output_var_names": output_var_names
    }

    stream_id = _generate_stream_id()
    for idx in range(map_block.desc.op_size()):
        map_block.desc.op(idx)._set_attr('_stream_id', stream_id)

    helper.append_op(
        type="map",
        inputs={"In": flat_inputs},
        outputs={"Out": flat_outputs},
        attrs=attrs)

    return outputs


def data_reader(reader_func,
                batch_size=1,
                num_samples=1,
                shuffle=False,
                drop_last=False,
                seed=None):
    assert not in_dygraph_mode(), \
            "paddle.io.data_reader can only be used in static mode"
    helper = LayerHelper("data_reader", **locals())

    # build reader block
    main_program = helper.main_program
    with _ProgramGuard(main_program):
        reader_block = main_program.current_block()

        indices_var = reader_block.create_var(
            name=unique_name.generate("data_reader_sub"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype="int64",
            persistable=False)
        program_outputs = reader_func(indices_var)
        program_outputs = _to_list(program_outputs)

        indices_var_name = indices_var.name
        output_var_names = []
        for outs in program_outputs:
            if isinstance(outs, (list, tuple)):
                for out in outs:
                    output_var_names.append(out.name)
            else:
                output_var_names.append(outs.name)

    outputs = []
    for outps in program_outputs:
        if isinstance(outps, (list, tuple)):
            for outp in outps:
                outputs.append(
                    helper.create_variable(
                        name=unique_name.generate("data_reader"),
                        type=outp.desc.type(),
                        dtype=outp.desc.dtype(),
                        persistable=True))
        else:
            outputs.append(
                helper.create_variable(
                    name=unique_name.generate("data_reader"),
                    type=outps.desc.type(),
                    dtype=outps.desc.dtype(),
                    persistable=True))

    attrs = {
        "reader_id": _hash_with_id(main_program),
        "reader_block": reader_block,
        "indices_var_name": indices_var_name,
        "output_var_names": output_var_names,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "seed": 0 if seed is None else seed,
        "rank": paddle.distributed.get_rank(),
        "world_size": paddle.distributed.get_world_size()
    }

    helper.append_op(
        type="data_reader", inputs={}, outputs={"Out": outputs}, attrs=attrs)

    return outputs
