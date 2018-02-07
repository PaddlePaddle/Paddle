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
    type=fluid.core.VarDesc.VarType.READER, name="RandomReader")
random_reader.desc.set_lod_levels([0, 0])

create_random_reader_op = block.append_op(
    type="create_random_reader",
    outputs={"Out": random_reader},
    attrs={
        "shape_concat": [1, 2, 1, 1],
        "ranks": [2, 2],
        "min": 0.0,
        "max": 1.0
    })

out1 = block.create_var(
    type=fluid.core.VarDesc.VarType.LOD_TENSOR,
    name="Out1",
    shape=[10, 2],
    dtype="float32",
    lod_level=1)
out2 = block.create_var(
    type=fluid.core.VarDesc.VarType.LOD_TENSOR,
    name="Out2",
    shape=[10, 1],
    dtype="float32",
    lod_level=1)

read_op = block.append_op(
    type="read",
    inputs={"Reader": random_reader},
    outputs={"Out": [out1, out2]})

place = fluid.CPUPlace()
exe = fluid.Executor(place)

[res1, res2] = exe.run(prog, fetch_list=[out1, out2])

if len(res1) == 0 or len(res2) == 0:
    exit(1)

exit(0)
