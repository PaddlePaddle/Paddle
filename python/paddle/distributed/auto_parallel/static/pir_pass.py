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

import paddle

from .process_group import new_process_group


def apply_partition_pass(program):
    new_program = program.clone()
    with paddle.static.program_guard(new_program):
        for op in new_program.global_block().ops:
            # assert len(op.operands()) == len(op.dist_attr().operand_dist_attrs()), f'The number of operand and operand_dist_attrs are not equal in op: {op}'
            for var, operand_dist_attr in zip(
                op.operands(), op.dist_attr().operand_dist_attrs()
            ):
                if (
                    var.source().is_dist_dense_tensor_type()
                    and var.source().dist_attr() != operand_dist_attr
                ):
                    paddle.pir.set_insertion_point(op)
                    # insert reshard
                    reshard_var = paddle._pir_ops.reshard_v2(
                        var.source(), operand_dist_attr
                    )
                    var.set_source(reshard_var)
    return new_program


def apply_reshard_pass(program):
    new_program = program.clone()
    with paddle.static.program_guard(new_program):
        for op in new_program.global_block().ops:
            # TODO(ywt): add common reshard rules
            # only support 1-D partial to replicated now
            if op.name() == 'dist_op.reshard':
                process_mesh = op.operand(0).source().dist_attr().process_mesh
                assert (
                    len(process_mesh.shape) == 1
                ), f'only support 1-D mesh now, but the op is: {op}'
                assert op.operand(0).source().dist_attr().partial_dims == {
                    0
                }, f'only support partial input on 1-D mesh now, but the op is: {op}'
                assert (
                    op.result(0).dist_attr().partial_dims == set()
                ), f'only support un-partial output on 1-D mesh now, but the op is: {op}'
                assert (
                    op.result(0).dist_attr().dims_mapping
                    == op.operand(0).source().dist_attr().dims_mapping
                ), f'only support the same dims maping on 1-D mesh now, but the op is: {op}'
                assert (
                    op.dist_attr().operand_dist_attr(0).partial_status[0]
                    == paddle.distributed.ReduceType.kRedSum
                ), f'only support partial sum now, but the op is: {op}'
                assert (
                    op.operand(0).source().has_one_use()
                ), f'only support use count of 1 for reshard input, but the op is: {op}'
                assert op.result(
                    0
                ).has_one_use(), f'only support use count of 1 for reshard output, but the op is: {op}'

                paddle.pir.set_insertion_point(op)
                group = new_process_group(process_mesh.process_ids)
                reduced_value = paddle._pir_ops.c_allreduce_sum_(
                    op.operand(0).source(), group.id, False, False
                )
                reduced_value.set_type(op.result(0).type())
                op.result(0).replace_all_uses_with(reduced_value)
                new_program.global_block().remove_op(op)

    return new_program
