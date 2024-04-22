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
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs

register_reshard_funcs()


def apply_partition_pass(program):
    new_program = program.clone()
    with paddle.static.program_guard(new_program):
        for op in new_program.global_block().ops:
            # assert len(op.operands()) == len(op.dist_attr().operand_dist_attrs()), f'The number of operand and operand_dist_attrs are not equal in op: {op}'
            for var, operand_dist_attr in zip(
                op.operands(), op.dist_attr.operand_dist_attrs()
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

        # pruning op and value not belong to cur rank
        cur_rank = paddle.distributed.get_rank()
        for op in new_program.global_block().ops[::-1]:
            if cur_rank not in op.dist_attr.process_mesh.process_ids:
                new_program.global_block().remove_op(op)
            else:
                # set the operand as null when it is not belong to cur rank
                if (
                    op.name() == 'dist_op.reshard'
                    and cur_rank
                    not in op.operand(0)
                    .source()
                    .dist_attr()
                    .process_mesh.process_ids
                ):
                    op.operand(0).set_source(paddle.pir.fake_value())

        # merge pd.data ops for
        lr_ops = []
        for op in new_program.global_block().ops[::-1]:
            if (
                op.name() == 'pd_op.data'
                and "learning_rate" in op.attrs()["name"]
            ):
                lr_ops.append(op)
        if len(lr_ops) > 1:
            lr_value = lr_ops[0].result(0)
            for op in lr_ops[1:]:
                lr = op.result(0)
                lr.replace_all_uses_with(lr_value)
                new_program.global_block().remove_op(op)

    return new_program


def apply_reshard_pass_deprecated(program):
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
                    op.dist_attr.operand_dist_attr(0).partial_status[0]
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


def apply_reshard_pass(program):
    new_program = program.clone()
    with paddle.base.program_guard(new_program):
        for op in new_program.global_block().ops:
            if op.name() == 'dist_op.reshard':
                op_dist_attr = op.attrs()["op_dist_attr"]
                src_dist_attr = op_dist_attr.operand_dist_attr(0)
                dst_dist_attr = op_dist_attr.result_dist_attr(0)

                reshard_func = choose_reshard_func(src_dist_attr, dst_dist_attr)
                assert (
                    reshard_func is not None
                ), f"Could not find reshard func for src {src_dist_attr}, dst {dst_dist_attr}"
                reshard_func.reshard(
                    new_program, op, src_dist_attr, dst_dist_attr
                )

    return new_program
