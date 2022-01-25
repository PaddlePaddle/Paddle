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
from ...common_ops_import import *


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


def map(map_func, inputs):
    inputs = _to_list(inputs)
    if in_dygraph_mode():
        return map_func(*inputs)

    helper = LayerHelper("map", **locals())

    # build map block
    main_program = helper.main_program
    with _ProgramGuard(main_program):
        program_id = _hash_with_id(main_program, map_func)
        map_block = main_program.current_block()

        program_inputs = [
            map_block.create_var(
                name=unique_name.generate("map_sub"),
                type=inputs[0].desc.type(),
                persistable=False) for inp in inputs]
        program_outputs = map_func(*program_inputs)
        program_outputs = _to_list(program_outputs)
    
        input_var_names = [v.name for v in program_inputs]
        output_var_names = [v.name for v in program_outputs]

    outputs = \
        [helper.create_variable(
            name=unique_name.generate("map"),
            type=outp.desc.type(),
            persistable=True) for outp in program_outputs]
    attrs = {
        "map_block": map_block,
        "program_id": program_id,
        "input_var_names": input_var_names,
        "output_var_names": output_var_names
    }

    stream_id = _generate_stream_id()
    for idx in range(map_block.desc.op_size()):
        map_block.desc.op(idx)._set_attr('stream_id', stream_id)

    helper.append_op(
        type="map",
        inputs={"In": inputs},
        outputs={"Out": outputs},
        attrs=attrs)

    return outputs


def data_reader(reader_func,
                batch_size=1,
                num_samples=1,
                shuffle=False,
                drop_last=False):
    assert not in_dygraph_mode(), \
            "paddle.io.data_reader can only be used in static mode"
    helper = LayerHelper("data_reader", **locals())

    # build reader block
    main_program = helper.main_program
    with _ProgramGuard(main_program):
        reader_id= _hash_with_id(main_program, reader_func)
        reader_block = main_program.current_block()

        indices_var = reader_block.create_var(
                        name=unique_name.generate("data_reader_sub"),
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        dtype="int64",
                        persistable=False)
        program_outputs = reader_func(indices_var)
        program_outputs = _to_list(program_outputs)
    
        indices_var_name = indices_var.name
        output_var_names = [v.name for v in program_outputs]

    outputs = \
        [helper.create_variable(
            name=unique_name.generate("data_reader"),
            type=outp.desc.type(),
            persistable=True) for outp in program_outputs]

    attrs = {
        "reader_id": reader_id,
        "reader_block": reader_block,
        "indices_var_name": indices_var_name,
        "output_var_names": output_var_names,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "rank": paddle.distributed.get_rank(),
        "world_size": paddle.distributed.get_world_size()
    }

    helper.append_op(
        type="data_reader",
        inputs={},
        outputs={"Out": outputs},
        attrs=attrs)

    return outputs
