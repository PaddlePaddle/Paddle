# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle
from paddle.fluid import unique_name, core
from paddle.fluid.framework import Program
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY
from paddle.distributed.fleet.meta_optimizers.sharding.utils import get_var_size

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

__CPU_VAR_SUFFIX__ = "@CUDAPINNED"


def current_allocated_mem_in_mb():
    return paddle.device.cuda.memory_allocated(0) / 1024.0 / 1024.0


def naive_offload_decorate(program, executor, scope=None):
    """
    Given a program, offload all its persistable variables to CPU (cuda pinned memory).

    The decorated program execution logic would be modified as following steps:

        * 1. Prefetch **all** persistable vars to GPU
        * 2. Run original program logic
        * 3. Release **all** fetched vars in GPU (there would be no variable hold GPU allocation in this program in ideal case)

    It is strongly recommanded that use a individual Scope for execution of decorated program.

    """

    logger.info("Offload Program: {} ... ".format(id(program)))

    # collect all offload vars
    logger.info("The following variable will be offload to CPU: ")
    vars_to_offload = []
    offload_size_mb = 0
    for var in program.list_vars():
        if (
            var.persistable
            and var.is_parameter
            and var.type == core.VarDesc.VarType.LOD_TENSOR
        ):
            vars_to_offload.append(var)
            var_size = get_var_size(var)
            offload_size_mb += var_size
            logger.info(
                "Offload  {}  shape: {}, size: {} MB.".format(
                    var.name, var.shape, var_size
                )
            )
    logger.info("Total {} MB variable Offloaded.".format(offload_size_mb))
    logger.info(
        "Before Offload, GPU Allocated Memory: {} MB.".format(
            current_allocated_mem_in_mb()
        )
    )

    # Offload vars to CPU
    offload_program = Program()
    offload_block = offload_program.global_block()
    gpu_var_name_to_cpu_var_names = {}
    for gpu_var in vars_to_offload:
        cpu_var_name = unique_name.generate(gpu_var.name + __CPU_VAR_SUFFIX__)
        gpu_var_name_to_cpu_var_names[gpu_var.name] = cpu_var_name

        # offload_block_gpu_var = offload_block.create_var(
        #     name=gpu_var.name,
        #     shape=gpu_var.shape,
        #     dtype=gpu_var.dtype,
        #     persistable=False,
        #     stop_gradient=True)

        offload_block_cpu_var = offload_block.create_var(
            name=cpu_var_name,
            shape=gpu_var.shape,
            dtype=gpu_var.dtype,
            persistable=True,
            stop_gradient=True,
        )

        offload_block.append_op(
            type='memcpy',
            inputs={'X': gpu_var},
            outputs={'Out': offload_block_cpu_var},
            attrs={"dst_place_type": 2, OP_ROLE_KEY: 0},
        )

    executor.run(offload_program, scope=scope)
    # TODO clear the GPU allocation of the gpu var
    logger.info(
        "After Offload, GPU Allocated Memory: {} MB.".format(
            current_allocated_mem_in_mb()
        )
    )

    # Prefecth variables
    # TODO support conditional blocks
    original_block = program.global_block()
    for gpu_var in vars_to_offload:

        cpu_var_name = gpu_var_name_to_cpu_var_names[gpu_var.name]
        cpu_var = offload_block.create_var(
            name=cpu_var_name,
            shape=gpu_var.shape,
            dtype=gpu_var.dtype,
            persistable=True,
            stop_gradient=True,
        )

        original_block._insert_op_without_sync(
            0,
            type='memcpy',
            inputs={'X': cpu_var},
            outputs={'Out': gpu_var},
            attrs={"dst_place_type": 1, OP_ROLE_KEY: 0},
        )

        gpu_var.persistable = False

    return program, offload_program
