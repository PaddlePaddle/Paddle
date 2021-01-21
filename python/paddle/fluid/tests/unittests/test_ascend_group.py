# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import time

def init_communicator(program, main_program, current_endpoint, endpoints, ring_id):
    nranks = len(endpoints)
    other_endpoints = endpoints[:]
    other_endpoints.remove(current_endpoint)
    group_rank=endpoints.index(current_endpoint)
    assert group_rank >=0

    block = program.global_block()
    nccl_id_var = block.create_var(
        name=unique_name.generate('nccl_id'),
        persistable=True,
        type=core.VarDesc.VarType.RAW)
    block.append_op(
        type='c_gen_nccl_id',
        inputs={},
        outputs={'Out': nccl_id_var},
        attrs={
            'rank': group_rank,
            'endpoint': current_endpoint,
            'other_endpoints': other_endpoints,
            OP_ROLE_KEY: OpRole.Forward,
        })
    block.append_op(
        type='c_comm_init',
        inputs={'X': nccl_id_var},
        outputs={},
        attrs={
            'nranks': nranks,
            'rank': group_rank,
            'ring_id': ring_id,
            OP_ROLE_KEY: OpRole.Forward,
        })
    block.create_var(
        name="data",
        persistable=True,
        dtype='float32')

    with fluid.program_guard(main_program):
        data=fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=2.5)
        helper.append_op(
            type=op_type,
            inputs={'X': [data]},
            outputs={'Out': [data]},
            attrs={'ring_id': ring_id,
                   'use_calc_stream': True})

def train():
    startup_programs=[]
    main_programs=[]

    trainer_endpoints=["127.0.0.1:6071","127.0.0.1:6072","127.0.0.1:6073","127.0.0.1:6074"]
    groups=[]
    groups[0]=[trainer_endpoints[0], trainer_endpoints[1]]
    groups[1]=[trainer_endpoints[2], trainer_endpoints[3]]
    groups[3]=[trainer_endpoints[0], trainer_endpoints[2]]

    for idx, group in enumerate(groups):
        for te in group:
            startup_program = fluid.program()
            init_communicator(startup_program, main_program, te, group, str(idx))
            startup_programs.append(startup_program)
            main_programs.append(main_program)

    print(startup_program[0])
    print(main_program[0])


