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

import logging

from paddle.base import core
from paddle.distributed.auto_parallel.static.operators.common import (
    is_data_parallel_broadcast_op,
    is_data_parallel_reduce_op,
    is_data_parallel_scale_op,
)

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    _create_param,
    _program_for_vpp,
    _program_for_vpp_split_bwk,
    split_matmul_grad_to_matmul,
)
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_VPP")
class PipelineVirtualPipelinePass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self._real_overlap_sharding_reduce = False
        self._real_overlap_sharding_broadcast = False
        self.reduce_comm_suffix = "_reduce"
        self.broadcast_comm_suffix = "_broadcast"
        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}

    def _record_fwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._forward_micro_step_counter[virtual_pp_rank]
        self._forward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _record_bwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._backward_micro_step_counter[virtual_pp_rank]
        self._backward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _create_job_list(self):
        accumulate_steps = self.get_attr("num_micro_batches")
        stage_id = self.get_attr("pp_stage")
        num_stages = self.get_attr("pp_degree")
        num_model_chunks = self.get_attr("vpp_degree")
        split_backward = self.get_attr("split_backward", False)
        for i in range(num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

        assert accumulate_steps % num_stages == 0

        def _get_virtual_pp_rank(micro_step, forward):
            virtual_pp_stage = micro_step % (num_stages * num_model_chunks)
            virtual_pp_stage = virtual_pp_stage // num_stages
            if not forward:
                virtual_pp_stage = num_model_chunks - virtual_pp_stage - 1
            return virtual_pp_stage

        total_num_steps = accumulate_steps * num_model_chunks
        if accumulate_steps == num_stages:
            warmup_steps = total_num_steps
        else:
            warmup_steps = (num_stages - stage_id - 1) * 2
            warmup_steps += (num_model_chunks - 1) * num_stages
            warmup_steps = min(warmup_steps, total_num_steps)

        steady_steps = total_num_steps - warmup_steps
        real_split_backward = (
            accumulate_steps == num_stages
        ) and split_backward

        job_list = []
        for micro_step in range(warmup_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=True)
            micro_batch_id = self._record_fwd_micro_step(virtual_pp_rank)
            if self._real_overlap_sharding_broadcast:
                if stage_id == 0 and micro_batch_id == 0:
                    fw_job = core.Job(
                        FORWARD
                        + str(virtual_pp_rank)
                        + self.broadcast_comm_suffix
                    )
                elif (
                    stage_id != 0
                    and micro_batch_id == 0
                    and virtual_pp_rank == 0
                ):
                    fw_job = core.Job(
                        FORWARD
                        + str(virtual_pp_rank)
                        + self.broadcast_comm_suffix
                    )
                else:
                    fw_job = core.Job(FORWARD + str(virtual_pp_rank))
            else:
                fw_job = core.Job(FORWARD + str(virtual_pp_rank))
            fw_job.set_micro_batch_id(micro_batch_id)
            job_list.append(fw_job)

        for micro_step in range(steady_steps):
            fwd_micro_step = micro_step + warmup_steps
            fwd_virtual_pp_rank = _get_virtual_pp_rank(
                fwd_micro_step, forward=True
            )
            fwd_micro_batch_id = self._record_fwd_micro_step(
                fwd_virtual_pp_rank
            )
            if self._real_overlap_sharding_broadcast:
                if stage_id == 0 and micro_batch_id == 0:
                    fw_job = core.Job(
                        FORWARD
                        + str(virtual_pp_rank)
                        + self.broadcast_comm_suffix
                    )
                else:
                    fwd_job = core.Job(FORWARD + str(fwd_virtual_pp_rank))
            else:
                fwd_job = core.Job(FORWARD + str(fwd_virtual_pp_rank))
            fwd_job.set_micro_batch_id(fwd_micro_batch_id)
            job_list.append(fwd_job)

            bw_micro_step = micro_step
            bwd_virtual_pp_rank = _get_virtual_pp_rank(
                bw_micro_step, forward=False
            )
            bwd_micro_batch_id = self._record_bwd_micro_step(
                bwd_virtual_pp_rank
            )
            if real_split_backward:
                bwd_job = core.Job(BACKWARD + "_b" + str(bwd_virtual_pp_rank))
            else:
                bwd_job = core.Job(BACKWARD + str(bwd_virtual_pp_rank))
            bwd_job.set_micro_batch_id(bwd_micro_batch_id)
            job_list.append(bwd_job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            if real_split_backward:
                bwd_job = core.Job(BACKWARD + "_b" + str(virtual_pp_rank))
            else:
                bwd_job = core.Job(BACKWARD + str(virtual_pp_rank))
            bwd_job.set_micro_batch_id(micro_batch_id)
            job_list.append(bwd_job)
            # TODO(lizhiyu): Inserting 'backward_b' and 'backward_w' interleavedly can decrease the memory,
            #                but it reduces the speed. We should find the better way to use the code here.
            # next_virtual_pp_rank = _get_virtual_pp_rank(micro_step + 1, forward=False)
            # if next_virtual_pp_rank != virtual_pp_rank:
            #     for micro_batch_id in range(0, accumulate_steps):
            #         w_job = core.Job(BACKWARD + "_w" + str(virtual_pp_rank))
            #         w_job.set_micro_batch_id(micro_batch_id)
            #         job_list.append(w_job)

        if real_split_backward:
            for chunk_id in range(num_model_chunks - 1, -1, -1):
                for micro_batch_id in range(0, accumulate_steps):
                    if (
                        self._real_overlap_sharding_reduce
                        and micro_batch_id == accumulate_steps - 1
                    ):
                        w_job = core.Job(
                            BACKWARD
                            + "_w"
                            + str(chunk_id)
                            + self.reduce_comm_suffix
                        )
                    else:
                        w_job = core.Job(BACKWARD + "_w" + str(chunk_id))
                    w_job.set_micro_batch_id(micro_batch_id)
                    job_list.append(w_job)
        job_types = [job.type() for job in job_list]
        logger.debug(f"The VPP job list: {job_types}")
        opt_job = core.Job(OPT)
        job_list.append(opt_job)
        return job_list

    def _split_matmul_grad_ops_to_matmul(self, program, dist_context):
        for block in program.blocks:
            matmul_grad_op_idx = []
            ops = block.ops
            for i, op_i in enumerate(ops):
                if (
                    op_i.type == "matmul_v2_grad"
                    and not op_i.attr("trans_x")
                    and not op_i.attr("trans_y")
                ):
                    matmul_grad_op_idx.append(i)

            for matmul_grad_id in reversed(matmul_grad_op_idx):
                split_matmul_grad_to_matmul(
                    block, matmul_grad_id, dist_context=dist_context
                )

    def _move_sharding_comm_to_backward(
        self, types, sub_programs, global_grads
    ):
        def _get_sharding_comm_op(op, idx, ops):
            if is_data_parallel_reduce_op(op):
                op_input_names = op.desc.input_arg_names()
                op_output_names = op.desc.output_arg_names()
                if (
                    op_input_names[0] == op_output_names[0]
                    and op_input_names[0] in global_grads
                ):
                    global_grad_to_comm_op[op_input_names[0]] = [op]
                    remove_op_ids.append(idx)

                if op.type in ["c_allreduce_sum", "c_reduce_sum"]:
                    scale_index = idx + 1
                    if scale_index < len(len(ops)):
                        if is_data_parallel_scale_op(ops[scale_index]):
                            global_grad_to_comm_op[op_input_names[0]].append(op)
                            remove_op_ids.append(scale_index)

        def _get_scale_op(op, idx):
            if is_data_parallel_scale_op(op):
                return
            if op.type == 'scale':
                op_input_names = op.desc.input_arg_names()
                op_output_names = op.desc.output_arg_names()
                if (
                    op_input_names[0] == op_output_names[0]
                    and op_input_names[0] in global_grads
                ):
                    global_grad_to_scale_op[op_input_names[0]] = op
                    remove_op_ids.append(idx)

        # 1 get the all sharding_avg in optimizer
        type_programs = dict(zip(types, sub_programs))
        opt_program = type_programs["optimizer"]
        global_grad_to_comm_op = {}
        global_grad_to_scale_op = {}
        all_remove_op_ids = []
        for cur_block in opt_program.blocks:
            remove_op_ids = []
            for idx, op in enumerate(cur_block.ops):
                _get_scale_op(op, idx)
                _get_sharding_comm_op(op, idx, cur_block.ops)
            all_remove_op_ids.append(remove_op_ids)
        if len(global_grad_to_comm_op) == 0:  # no need to overlap sharding comm
            return False

        # 2 create the new backward(w) with the sharding_comm
        new_types = []
        new_programs = []
        for type, sub_program in type_programs.items():
            if "backward_w" in type:
                new_program = sub_program.clone()
                cur_block = new_program.global_block()
                cur_block_scale_op = []
                for idx, op in reversed(list(enumerate(cur_block.ops))):
                    if op.type == "elementwise_add":
                        input_arg_names = op.input_arg_names
                        output_arg_names = op.output_arg_names
                        if (
                            input_arg_names[0] == output_arg_names[0]
                            and input_arg_names[0] in global_grad_to_comm_op
                        ):
                            for origin_op in reversed(
                                global_grad_to_comm_op[input_arg_names[0]]
                            ):
                                new_op = cur_block._insert_op_without_sync(
                                    index=idx + 1, type="nop"
                                )
                                new_op.desc.copy_from(origin_op.desc)
                            del global_grad_to_comm_op[input_arg_names[0]]
                            cur_block_scale_op.append(
                                global_grad_to_scale_op[input_arg_names[0]]
                            )
                for origin_op in cur_block_scale_op:
                    new_op = cur_block.append_op(type="nop")
                    new_op.desc.copy_from(origin_op.desc)
                cur_block._sync_with_cpp()
                new_types.append(type + self.reduce_comm_suffix)
                new_programs.append(new_program)
        assert (
            len(global_grad_to_comm_op) == 0
        ), f"global_grad_to_comm_op must be used up, but left: {global_grad_to_comm_op}"

        types.extend(new_types)
        sub_programs.extend(new_programs)

        for id, cur_block in enumerate(opt_program.blocks):
            for op_id in reversed(all_remove_op_ids[id]):
                cur_block._remove_op(op_id)
            cur_block._sync_with_cpp()

        return True

    def _move_sharding_broadcast_to_backward(
        self, types, sub_programs, stage_id
    ):
        def _get_sharding_broadcast_op(op, idx, cur_block):
            if is_data_parallel_broadcast_op(op):
                op_input_names = op.desc.input_arg_names()
                op_output_names = op.desc.output_arg_names()
                if op_input_names[0] == op_output_names[0]:
                    param_to_comm_op[op_input_names[0]] = op
                    param_name_to_params[
                        op_input_names[0]
                    ] = cur_block._var_recursive(op_input_names[0])
                    remove_op_ids.append(idx)

        # 1 get the all sharding_avg in optimizer
        type_programs = dict(zip(types, sub_programs))
        opt_program = type_programs["optimizer"]
        param_to_comm_op = {}
        param_name_to_params = {}
        all_remove_op_ids = []
        for cur_block in opt_program.blocks:
            remove_op_ids = []
            for idx, op in enumerate(cur_block.ops):
                _get_sharding_broadcast_op(op, idx, cur_block)
            all_remove_op_ids.append(remove_op_ids)
        if len(param_to_comm_op) == 0:  # no need to overlap sharding comm
            return False

        # 2 create the new forward with the sharding_broadcast
        new_types = []
        new_programs = []
        sub_program_ids = []
        sub_program_count = 0
        for type, sub_program in type_programs.items():
            if stage_id == 0:
                if "forward" in type:
                    sub_program_ids.append(sub_program_count)
                    new_program = sub_program.clone()
                    cur_block = new_program.global_block()
                    insert_op_ids = []
                    insert_ops = []
                    for idx, op in enumerate(cur_block.ops):
                        input_arg_names = op.input_arg_names
                        for input_name in input_arg_names:
                            if input_name in param_to_comm_op:
                                # NOTE(lizhiyu): The number that we insert 'broadcast' before the operator using the parameter
                                #               may be different in different case.
                                insert_op_ids.append(max(0, idx - 3))
                                insert_ops.append(param_to_comm_op[input_name])
                                del param_to_comm_op[input_name]
                    for op_id, origin_op in zip(
                        reversed(insert_op_ids), reversed(insert_ops)
                    ):
                        new_op = cur_block._insert_op_without_sync(
                            index=op_id, type="nop"
                        )
                        new_op.desc.copy_from(origin_op.desc)
                        new_op.dist_attr.execution_stream = (
                            "sharding_comm_broadcast_stream"
                        )
                    cur_block._sync_with_cpp()
                    new_types.append(type + self.broadcast_comm_suffix)
                    new_programs.append(new_program)
                sub_program_count += 1
            else:
                if "forward0" in type:
                    sub_program_ids.append(sub_program_count)
                    new_program = sub_program.clone()
                    cur_block = new_program.global_block()
                    insert_op_ids = range(0, len(param_to_comm_op))
                    insert_ops = list(param_to_comm_op.values())
                    for op_id, origin_op in zip(insert_op_ids, insert_ops):
                        new_op = cur_block._insert_op_without_sync(
                            index=op_id, type="nop"
                        )
                        new_op.desc.copy_from(origin_op.desc)
                        new_op.dist_attr.execution_stream = (
                            "sharding_comm_broadcast_stream"
                        )

                        # create var
                        input_arg_names = new_op.input_arg_names
                        for input_name in input_arg_names:
                            if not cur_block.has_var(input_name):
                                src_var = param_name_to_params[input_name]
                                _create_param(cur_block, src_var)

                    cur_block._sync_with_cpp()
                    new_types.append(type + self.broadcast_comm_suffix)
                    new_programs.append(new_program)
                sub_program_count += 1
        if stage_id == 0:
            assert (
                len(param_to_comm_op) == 0
            ), f"param_to_comm_op must be used up, but left: {param_to_comm_op}"

        for i in range(len(sub_program_ids) - 1, -1, -1):
            program_idx = sub_program_ids[i]
            types.insert(program_idx, new_types[i])
            sub_programs.insert(program_idx, new_programs[i])

        for id, cur_block in enumerate(opt_program.blocks):
            for op_id in reversed(all_remove_op_ids[id]):
                cur_block._remove_op(op_id)
            cur_block._sync_with_cpp()
        return True

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        num_model_chunks = self.get_attr("vpp_degree")
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        accumulate_steps = self.get_attr("num_micro_batches")
        num_stages = self.get_attr("pp_degree")
        stage_id = self.get_attr("pp_stage")
        sharding_stage = self.get_attr("sharding_stage")
        split_backward = self.get_attr("split_backward", False)
        grad_to_global_grad = self.get_attr("grad_to_global_grad", {})
        global_grads = [
            global_grad for _, global_grad in grad_to_global_grad.items()
        ]
        if split_backward and accumulate_steps == num_stages:
            self._split_matmul_grad_ops_to_matmul(program, dist_context)
            types, sub_program_list = _program_for_vpp_split_bwk(
                program,
                num_model_chunks,
                dist_context,
                enable_send_recv_overlap,
            )
            self._real_overlap_sharding_reduce = (
                self._move_sharding_comm_to_backward(
                    types, sub_program_list, global_grads
                )
            )
        else:
            types, sub_program_list = _program_for_vpp(
                program,
                num_model_chunks,
                dist_context,
                enable_send_recv_overlap,
            )
        if sharding_stage in [1, 2]:
            self._real_overlap_sharding_broadcast = (
                self._move_sharding_broadcast_to_backward(
                    types=types,
                    sub_programs=sub_program_list,
                    stage_id=stage_id,
                )
            )

        return types, sub_program_list
