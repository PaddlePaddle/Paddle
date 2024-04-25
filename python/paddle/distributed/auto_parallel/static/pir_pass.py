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

from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs

register_reshard_funcs()


def apply_partition_pass(program):
    new_program = program.clone()
    with paddle.static.program_guard(new_program):
        for op in new_program.global_block().ops:
            assert len(op.operands()) == len(
                op.dist_attr.operand_dist_attrs()
            ), f'The number of operand and operand_dist_attrs are not equal in op: {op}'

            for var, operand_dist_attr in zip(
                op.operands(), op.dist_attr.operand_dist_attrs()
            ):
                prev_var = var.source()
                if (
                    prev_var.is_dist()
                    and prev_var.dist_attr() != operand_dist_attr
                ):
                    paddle.pir.set_insertion_point(op)
                    # fold reshard
                    if prev_var.get_defining_op().name() == 'dist_op.reshard':
                        prev_reshard = prev_var.get_defining_op()
                        prev_var = prev_reshard.operand_source(0)
                        if prev_var.dist_attr() == operand_dist_attr:
                            var.set_source(prev_var)
                        else:
                            reshard_var = paddle._C_ops.reshard_v2(
                                prev_var, operand_dist_attr
                            )
                            var.set_source(reshard_var)
                        if prev_reshard.result(0).use_empty():
                            prev_reshard.get_parent_block().remove_op(
                                prev_reshard
                            )
                        continue
                    # insert reshard
                    reshard_var = paddle._C_ops.reshard_v2(
                        prev_var, operand_dist_attr
                    )
                    var.set_source(reshard_var)

            for var, result_dist_attr in zip(
                op.results(), op.dist_attr.result_dist_attrs()
            ):
                if var.initialized() and var.dist_attr() != result_dist_attr:
                    paddle.pir.set_insertion_point_after(op)
                    old_dist_attr = var.dist_attr()
                    var.update_dist_attr(result_dist_attr)
                    # insert reshard
                    reshard_var = paddle._C_ops.reshard_v2(var, old_dist_attr)
                    var.replace_all_uses_with(reshard_var)
                    reshard_var.get_defining_op().operand(0).set_source(var)

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


def apply_reshard_pass(program):
    new_program = program.clone()
    with paddle.base.program_guard(new_program):
        for op in new_program.global_block().ops:
            if op.name() == 'dist_op.reshard':
                var = op.operand(0)
                op_dist_attr = op.attrs()["op_dist_attr"]
                src_dist_attr = op_dist_attr.operand_dist_attr(0)
                dst_dist_attr = op_dist_attr.result_dist_attr(0)
                assert (
                    paddle.pir.is_fake_value(var.source())
                    or var.source().dist_attr() == src_dist_attr
                ), f"The dist_attr of reshard op's input and operand should be equal, but got {var.source().dist_attr()} and {src_dist_attr}"

                reshard_func = choose_reshard_func(src_dist_attr, dst_dist_attr)
                assert (
                    reshard_func is not None
                ), f'There is no reshard function that matches src_dist_attr: {src_dist_attr} and dst_dist_attr: {dst_dist_attr}'
                reshard_func.reshard(
                    new_program, op, src_dist_attr, dst_dist_attr
                )

    return new_program
