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


class VarGroup:
    def __init__(self, max_size):
        self.max_size = max_size
        self.dtype = None
        self.ring_id = -1
        self.root_id = -1
        self.numel = 0
        self.vars = []
        self.outputs = []
        self.coalesce_var = None
        self.coalesce_op_idx = None
        self.reduce_ops = []

    def acceptable(self, param, ring_id, root_id=-1):
        if self.numel == 0:
            return True
        else:
            if param.dtype != self.dtype:
                return False
            if ring_id != self.ring_id:
                return False
            if root_id != -1 and self.root_id != root_id:
                return False
            if self.numel + param.numel() > self.max_size:
                return False
            return True

    def collect(self, param, output, ring_id, root_id=-1):
        self.dtype = param.dtype
        self.ring_id = ring_id
        self.root_id = root_id
        self.numel += param.numel()
        self.vars.append(param)
        self.outputs.append(output)

    def __len__(self):
        return len(self.vars)


def apply_tensor_fusion_pass(
    program, bucket_size=210355872, sharding_ring_id=None
):
    new_program = program.clone()

    def group_grads(block, block_groups):
        visited_vars = []
        var_groups = []
        var_groups.append(VarGroup(bucket_size))
        for idx, op in enumerate(block.ops):
            if op.name() == "pd_op.if":
                group_grads(op.as_if_op().true_block(), block_groups)
                group_grads(op.as_if_op().false_block(), block_groups)
            if op.name() in ["pd_op.c_reduce_avg_"]:
                pre_var = op.operand_source(0)
                res_var = op.result(0)
                # if pre_var in visited_vars:
                #     continue
                append_into_group = False
                for idx, cur_group in enumerate(var_groups):
                    if cur_group.acceptable(
                        pre_var, op.attrs()["ring_id"], op.attrs()["root_id"]
                    ):
                        cur_group.collect(
                            pre_var,
                            res_var,
                            op.attrs()["ring_id"],
                            op.attrs()["root_id"],
                        )
                        cur_group.reduce_ops.append(op)
                        visited_vars.append(pre_var)
                        append_into_group = True
                        break
                if not append_into_group:
                    new_group = VarGroup(max_size=bucket_size)
                    new_group.collect(
                        pre_var,
                        res_var,
                        op.attrs()["ring_id"],
                        op.attrs()["root_id"],
                    )
                    visited_vars.append(pre_var)
                    new_group.reduce_ops.append(op)
                    var_groups.append(new_group)
        deleted_idx = []
        for idx, var_group in enumerate(var_groups):
            if len(var_group) <= 1:
                deleted_idx.append(idx)
        for idx in reversed(deleted_idx):
            var_groups.pop(idx)
        if len(var_groups) > 0:
            block_groups.append((block, var_groups))

    def deal_block(block):
        block_groups = []
        group_grads(block, block_groups)
        if len(block_groups) == 0:
            return
        for block, var_groups in block_groups:
            for var_group in var_groups:
                inserted_coalesce = False
                for comm_op in var_group.reduce_ops:
                    if not inserted_coalesce:
                        inserted_coalesce = True
                        paddle.pir.set_insertion_point(comm_op)
                        (
                            split_output,
                            fused_output,
                        ) = paddle._C_ops.coalesce_tensor_(
                            var_group.vars,
                            var_group.dtype,
                            True,
                            False,
                            False,
                            0.0,
                            True,
                            -1,
                            -1,
                            [],
                            [],
                        )
                        fused_output.persistable = True
                        for origin_var, new_var in zip(
                            var_group.outputs, split_output
                        ):
                            new_var.persistable = True
                            origin_var.replace_all_uses_with(new_var)
                            # new_var.get_defining_op().operand(0).set_source(origin_var)

                        # insert comm_op
                        fused_avg_grad = paddle._C_ops.c_reduce_avg_(
                            fused_output,
                            var_group.ring_id,
                            comm_op.attrs()["root_id"],
                            True,
                        )
                        fused_avg_grad.get_defining_op().set_execution_stream(
                            "sharding_op_comm0"
                        )
                        # delete comm_op
                        block.remove_op(comm_op)
                    else:
                        block.remove_op(comm_op)

    with paddle.pir_utils.IrGuard():
        with paddle.pir.core.program_guard(new_program):
            deal_block(new_program.global_block())
    paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": True})
    return new_program
