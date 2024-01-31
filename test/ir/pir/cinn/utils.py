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

from collections import defaultdict

import numpy as np

import paddle

FUSION_KEY_STR = "fusion_number"
__FUSION_KERNEL_NAME = "cinn_runtime.jit_kernel"
__IF_OP_NAME = "pd_op.if"
__WHILE_OP_NAME = "pd_op.while"


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


def get_fusion_number(block):
    fusion_number = 0
    for op in block.ops:
        op_name = op.name()
        if op_name == __FUSION_KERNEL_NAME:
            fusion_number += 1
        elif op_name == __IF_OP_NAME:
            fusion_number = (
                fusion_number
                + get_fusion_number(op.true_block())
                + get_fusion_number(op.false_block())
            )
        elif op_name == __WHILE_OP_NAME:
            fusion_number += get_fusion_number(op.body())

    return fusion_number


def check_fusion_number(static_fn, expected_number):
    """
    Check whether total number of __FUSION_KERNEL_NAME in Program
    is equal to expected_number.
    """
    program = get_pir_program(static_fn)
    fusion_number = get_fusion_number(program.global_block())
    np.testing.assert_equal(fusion_number, expected_number)


def get_fusion_structure_helper(block, map_info):
    if_op_idx, while_op_idx = 0, 0
    for op in block.ops:
        op_name = op.name()
        if op_name == __FUSION_KERNEL_NAME:
            map_info[FUSION_KEY_STR] += 1
        elif op_name == __IF_OP_NAME:
            true_key = f"if_{if_op_idx}"
            map_info[true_key] = {}
            get_fusion_structure_helper(op.true_block(), map_info[true_key])

            false_key = f"else_{if_op_idx}"
            get_fusion_structure_helper(op.true_block(), map_info[false_key])
            if_op_idx += 1
        elif op.name() == __WHILE_OP_NAME:
            key = f"while_{while_op_idx}"
            map_info[key] = {}
            get_fusion_structure_helper(op.body(), map_info[key])
            while_op_idx += 1


def get_fusion_structure(static_fn):
    program = get_pir_program(static_fn)
    map_info = defaultdict(int)
    get_fusion_structure_helper(program.global_block(), map_info)
    return dict(map_info)


def check_fusion_structure(static_fn, expected_structure):
    """
    Check whether fuse subgraph structre in Program is same with expected_structure.
    For examaple:
    expected_structure = {
        FUSION_KEY_STR: 3,
        "if_0": {
            FUSION_KEY_STR: 1
        }
        "else_0": {
            FUSION_KEY_STR: 1
        }
         "if_1": {
            FUSION_KEY_STR: 0
        }
        "else_1": {
            FUSION_KEY_STR: 0
        }

        "while_0":{
            FUSION_KEY_STR: 2
        }
    }
    """
    map_info = get_fusion_structure(static_fn)
    np.testing.assert_equal(map_info, expected_structure)
