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

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import numpy as np

prog = fluid.framework.Program()
block = prog.current_block()

random_reader = block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="RandomDataGenerator")
random_reader.desc.set_dtypes(
    [fluid.core.VarDesc.VarType.FP32, fluid.core.VarDesc.VarType.FP32])

create_random_data_generator_op = block.append_op(
    type="create_random_data_generator",
    outputs={"Out": random_reader},
    attrs={
        "shape_concat": [1, 2, 1, 1],
        "ranks": [2, 2],
        "min": 0.0,
        "max": 1.0,
        'lod_levels': [0, 0]
    })
shuffle_reader = block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="ShuffleReader")

create_shuffle_reader_op = block.append_op(
    type="create_shuffle_reader",
    inputs={"UnderlyingReader": random_reader},
    outputs={"Out": shuffle_reader},
    attrs={"buffer_size": 7})

batch_reader = block.create_var(
    type=fluid.core.VarDesc.VarType.READER, name="BatchReader")

create_batch_reader_op = block.append_op(
    type="create_batch_reader",
    inputs={"UnderlyingReader": shuffle_reader},
    outputs={"Out": batch_reader},
    attrs={"batch_size": 10})

out1 = block.create_var(type=fluid.core.VarDesc.VarType.LOD_TENSOR, name="Out1")
out2 = block.create_var(type=fluid.core.VarDesc.VarType.LOD_TENSOR, name="Out2")

read_op = block.append_op(
    type="read", inputs={"Reader": batch_reader},
    outputs={"Out": [out1, out2]})

place = fluid.CPUPlace()
exe = fluid.Executor(place)

[res1, res2] = exe.run(prog, fetch_list=[out1, out2])

if not (res1.shape == (10, 2) and res2.shape == (10, 1)):
    exit(1)

exit(0)
