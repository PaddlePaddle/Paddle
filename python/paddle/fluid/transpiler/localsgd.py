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

import sys
from paddle import fluid


def init_nccl(trainer_id, trainer_endpoints, current_endpoint, startup_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = 'nccl'
    t = fluid.DistributeTranspiler(config=config)

    t.transpile(
        trainer_id,
        trainers=trainer_endpoints,
        current_endpoint=current_endpoint,
        startup_program=startup_prog)


def gen_sync_program_simple(ntrainers, start_prog, main_prog):
    sync_prog = main_prog.clone()
    for var in start_prog.list_vars():
        if isinstance(var, fluid.framework.Parameter):
            snapshot_name = var.name + '@SNAPSHOT'

            block = start_prog.current_block()
            snapshot = block.create_var(
                name=snapshot_name,
                shape=var.shape,
                persistable=True,
                stop_gradient=True)
            block.append_op(
                type='assign', inputs={'X': [var]},
                outputs={'Out': [snapshot]})

            block = sync_prog.current_block()
            snapshot = block.create_var(
                name=snapshot_name,
                shape=var.shape,
                persistable=True,
                stop_gradient=True)
            block.append_op(
                type='elementwise_sub',
                inputs={'X': [snapshot],
                        'Y': [var]},
                outputs={'Out': [var]})
            block.append_op(type='sync_stream', attrs={'sync_type': 1})
            block.append_op(
                type='allreduce',
                inputs={'X': [var]},
                outputs={'Out': [var]},
                attrs={'reduce_type': 0})
            block.append_op(type='sync_stream', attrs={'sync_type': 2})
            block.append_op(
                type='scale',
                inputs={'X': [var]},
                outputs={'Out': [var]},
                attrs={'scale': 1.0 / ntrainers})
            block.append_op(
                type='elementwise_sub',
                inputs={'X': [snapshot],
                        'Y': [var]},
                outputs={'Out': [var]})
            block.append_op(
                type='assign', inputs={'X': [var]},
                outputs={'Out': [snapshot]})
    return sync_prog


def is_optimizer_op(op):
    if "Param" in op.input_names and \
            "LearningRate" in op.input_names:
        return True
    return False


def gen_sync_program(ntrainers, start_prog, main_prog):
    snapshot_key = '@SNAPSHOT'
    param_map = {}
    for var in start_prog.list_vars():
        if isinstance(var, fluid.framework.Parameter):
            snapshot_name = var.name + snapshot_key

            block = start_prog.current_block()
            snapshot = block.create_var(
                name=snapshot_name,
                shape=var.shape,
                persistable=True,
                stop_gradient=True)
            block.append_op(
                type='assign', inputs={'X': [var]},
                outputs={'Out': [snapshot]})

            param_map[var.name] = var

    sync_prog = main_prog.clone()
    block = sync_prog.global_block()
    snapshot_map = {}
    idx2op = []
    for i, op in enumerate(block.ops):
        if is_optimizer_op(op):
            idx2op.append((i, op, op.input('Param')[0]))

    for elem in reversed(idx2op):
        var = param_map[elem[2]]
        snapshot_name = var.name + snapshot_key
        snapshot = block.create_var(
            name=snapshot_name,
            shape=var.shape,
            persistable=True,
            stop_gradient=True)
        snapshot_map[var.name] = snapshot
        block._insert_op(
            elem[0] + 1,
            type='elementwise_sub',
            inputs={'X': [snapshot],
                    'Y': [var]},
            outputs={'Out': [var]})
        block._insert_op(
            elem[0] + 2, type='sync_stream', attrs={'sync_type': 1})
        block._insert_op(
            elem[0] + 3,
            type='allreduce',
            inputs={'X': [var]},
            outputs={'Out': [var]},
            attrs={'reduce_type': 0})

    block.append_op(type='sync_stream', attrs={'sync_type': 2})

    for elem in idx2op:
        var = param_map[elem[2]]
        snapshot = snapshot_map[elem[2]]
        block.append_op(
            type='scale',
            inputs={'X': [var]},
            outputs={'Out': [var]},
            attrs={'scale': 1.0 / ntrainers})
        block.append_op(
            type='elementwise_sub',
            inputs={'X': [snapshot],
                    'Y': [var]},
            outputs={'Out': [var]})
        block.append_op(
            type='assign', inputs={'X': [var]}, outputs={'Out': [snapshot]})

    return sync_prog
