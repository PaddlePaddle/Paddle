#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import sys

startup_prog = fluid.framework.Program()
startup_block = startup_prog.current_block()

random_reader = startup_block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="RandomDataGenerator")
random_reader.desc.set_dtypes(
    [fluid.core.VarDesc.VarType.FP32, fluid.core.VarDesc.VarType.FP32])
random_reader.persistable = True
shuffle_reader = startup_block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="ShuffleReader")
shuffle_reader.persistable = True
batch_reader = startup_block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="BatchReader")
batch_reader.persistable = True
double_buffer = startup_block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="DoubleBuffer")
double_buffer.persistable = True

main_prog = startup_prog.clone()
main_block = main_prog.current_block()

create_random_data_generator_op = startup_block.append_op(
    type="create_random_data_generator",
    outputs={"Out": random_reader},
    attrs={
        "shape_concat": [1, 2, 1, 1],
        "ranks": [2, 2],
        "low": 0.0,
        "high": 1.0,
        'lod_levels': [0, 0]
    })

create_shuffle_reader_op = startup_block.append_op(
    type="create_shuffle_reader",
    inputs={"UnderlyingReader": random_reader},
    outputs={"Out": shuffle_reader},
    attrs={"buffer_size": 7})

create_batch_reader_op = startup_block.append_op(
    type="create_batch_reader",
    inputs={"UnderlyingReader": shuffle_reader},
    outputs={"Out": batch_reader},
    attrs={"batch_size": 10})

create_double_buffer_reader_op = startup_block.append_op(
    type="create_double_buffer_reader",
    inputs={"UnderlyingReader": batch_reader},
    outputs={"Out": double_buffer})

out1 = main_block.create_var(
    type=fluid.core.VarDesc.VarType.LOD_TENSOR, name="Out1")
out2 = main_block.create_var(
    type=fluid.core.VarDesc.VarType.LOD_TENSOR, name="Out2")

main_block.var("DoubleBuffer").desc.set_shapes(double_buffer.desc.shapes())
main_block.var("DoubleBuffer").desc.set_dtypes(double_buffer.desc.dtypes())
main_block.var("DoubleBuffer").desc.set_lod_levels(
    double_buffer.desc.lod_levels())

read_op = main_block.append_op(
    type="read",
    inputs={"Reader": double_buffer},
    outputs={"Out": [out1, out2]})

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(startup_prog)

for i in range(1, 100):
    [res1, res2] = exe.run(main_prog, fetch_list=[out1, out2])
    if not (res1.shape == (10, 2) and res2.shape == (10, 1)):
        exit(1)
