# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict

import numpy as np

import paddle

JIT_KERNEL_NAME = "jit_kernel"
__IF_OP_NAME = "pd_op.if"
__WHILE_OP_NAME = "pd_op.while"


def unittest_use_cinn():
    use_cinn = os.getenv("FLAGS_pd_unittest_use_cinn", False)
    true_value_set = {True, 1, "1", "True", "true"}
    false_value_set = {False, 0, "0", "False", "false"}
    assert use_cinn in (true_value_set | false_value_set)
    return use_cinn in true_value_set


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


def get_pir_program(static_fn):
    assert hasattr(static_fn, "program_cache")
    runnable_program = static_fn.program_cache.last()[1][1].program
    return runnable_program.forward_program


def get_jit_kernel_number(block):
    jit_kernel_number = 0
    for op in block.ops:
        op_name = op.name()
        if JIT_KERNEL_NAME in op_name:
            jit_kernel_number += 1
        elif op_name == __IF_OP_NAME:
            jit_kernel_number = (
                jit_kernel_number
                + get_jit_kernel_number(op.as_if_op().true_block())
                + get_jit_kernel_number(op.as_if_op().false_block())
            )

    return jit_kernel_number


def check_jit_kernel_number(static_fn, expected_number):
    """
    Check whether total number of JIT_KERNEL_NAME in Program
    is equal to expected_number.
    """
    program = get_pir_program(static_fn)
    jit_kernel_number = get_jit_kernel_number(program.global_block())
    np.testing.assert_equal(jit_kernel_number, expected_number)


def get_jit_kernel_structure_helper(block, map_info, if_op_idx='_0'):
    """
    Recursivly generate JIT_KERNEL map_info for Static/Dynmaic Shape UT.
    """
    if_count = 0
    for op in block.ops:
        op_name = op.name()
        if JIT_KERNEL_NAME in op_name:
            if JIT_KERNEL_NAME not in map_info:
                map_info[JIT_KERNEL_NAME] = 0
            map_info[JIT_KERNEL_NAME] += 1
        elif op_name == __IF_OP_NAME:
            true_key = f"if{if_op_idx}"
            false_key = f"else{if_op_idx}"
            map_info[true_key] = {}
            map_info[false_key] = {}
            get_jit_kernel_structure_helper(
                op.as_if_op().true_block(),
                map_info[true_key],
                if_op_idx + '_' + str(if_count),
            )
            get_jit_kernel_structure_helper(
                op.as_if_op().false_block(),
                map_info[false_key],
                if_op_idx + '_' + str(if_count),
            )
            if_count += 1


def get_jit_kernel_structure(static_fn):
    program = get_pir_program(static_fn)
    map_info = defaultdict(int)
    get_jit_kernel_structure_helper(program.global_block(), map_info)
    return dict(map_info)


def check_jit_kernel_structure(static_fn, expected_structure):
    """
    Check whether fuse subgraph structure in Program is same with expected_structure.
    For examaple:
    expected_structure = {
        JIT_KERNEL_NAME: 3,
        "if_0": {
            JIT_KERNEL_NAME: 1
        }
        "else_0": {
            JIT_KERNEL_NAME: 1
        }
         "if_1": {
            JIT_KERNEL_NAME: 0
        }
        "else_1": {
            JIT_KERNEL_NAME: 0
        }

        "while_0":{
            JIT_KERNEL_NAME: 2
        }
    }
    """
    map_info = get_jit_kernel_structure(static_fn)
    np.testing.assert_equal(map_info, expected_structure)
