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
from collections import OrderedDict

import paddle
from paddle.base import core

from ...auto_parallel.static.utils import OpRole
from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    _create_program_and_ops,
    _get_device,
    _pir_get_backward_op_type,
    _pir_overlap_send_recv,
    _pir_split_matmul_grad_to_matmul,
    infer_chunk_id,
)
from .pipeline_pass_base import PipelinePassBase

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_VPP")
class PipelineVirtualPipelinePass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self._real_overlap_sharding_reduce = False
        self.reduce_comm_suffix = "_reduce"
        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}
        self.jobs_in_stable_phase_in_pir = [
            self.BACKWARD,
            self.RECV_FORWARD,
            self.SEND_BACKWARD,
            self.FORWARD,
        ]

    def _record_fwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._forward_micro_step_counter[virtual_pp_rank]
        self._forward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _record_bwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._backward_micro_step_counter[virtual_pp_rank]
        self._backward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _create_job_list(self):
        if self._in_pir_mode:
            return self._pir_create_job_list()
        accumulate_steps = self.get_attr("num_micro_batches")
        stage_id = self.get_attr("pp_stage")
        num_stages = self.get_attr("pp_degree")
        num_model_chunks = self.get_attr("vpp_degree")
        split_backward = self.get_attr("split_backward", False)
        remainder = accumulate_steps % num_stages
        for i in range(num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

        assert accumulate_steps >= num_stages

        def _get_virtual_pp_rank(micro_step, forward):
            virtual_pp_stage = micro_step % (num_stages * num_model_chunks)
            if micro_step <= (accumulate_steps // num_stages) * (
                num_stages * num_model_chunks
            ):
                virtual_pp_stage = virtual_pp_stage // num_stages
            else:
                virtual_pp_stage = virtual_pp_stage // remainder
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
            fw_job = core.Job(self.FORWARD + str(virtual_pp_rank))
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
            fwd_job = core.Job(self.FORWARD + str(fwd_virtual_pp_rank))
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
                bwd_job = core.Job(
                    self.BACKWARD + "_b" + str(bwd_virtual_pp_rank)
                )
            else:
                bwd_job = core.Job(self.BACKWARD + str(bwd_virtual_pp_rank))
            bwd_job.set_micro_batch_id(bwd_micro_batch_id)
            job_list.append(bwd_job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            if real_split_backward:
                bwd_job = core.Job(self.BACKWARD + "_b" + str(virtual_pp_rank))
            else:
                bwd_job = core.Job(self.BACKWARD + str(virtual_pp_rank))
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
                            self.BACKWARD
                            + "_w"
                            + str(chunk_id)
                            + self.reduce_comm_suffix
                        )
                    else:
                        w_job = core.Job(self.BACKWARD + "_w" + str(chunk_id))
                    w_job.set_micro_batch_id(micro_batch_id)
                    job_list.append(w_job)
        job_types = [job.type() for job in job_list]
        logger.debug(f"The VPP job list: {job_types}")
        opt_job = core.Job(self.OPT)
        job_list.append(opt_job)
        return job_list

    def _pir_create_job_list(self):
        accumulate_steps = self.get_attr("num_micro_batches")
        stage_id = self.get_attr("pp_stage")
        num_stages = self.get_attr("pp_degree")
        num_model_chunks = self.get_attr("vpp_degree")
        split_backward = self.get_attr("split_backward", False)
        remainder = accumulate_steps % num_stages
        for i in range(num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

        assert accumulate_steps >= num_stages

        def _get_virtual_pp_rank(micro_step, forward):
            virtual_pp_stage = micro_step % (num_stages * num_model_chunks)
            if micro_step <= (accumulate_steps // num_stages) * (
                num_stages * num_model_chunks
            ):
                virtual_pp_stage = virtual_pp_stage // num_stages
            else:
                virtual_pp_stage = virtual_pp_stage // remainder
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

        real_split_backward = (
            accumulate_steps == num_stages
        ) and split_backward
        if not real_split_backward:
            warmup_steps = min(total_num_steps, warmup_steps + 1)
        steady_steps = total_num_steps - warmup_steps
        job_list = []
        for micro_step in range(warmup_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=True)
            micro_batch_id = self._record_fwd_micro_step(virtual_pp_rank)
            if not real_split_backward:
                recv_fwd_job = core.Job(
                    self.RECV_FORWARD + str(virtual_pp_rank)
                )
                recv_fwd_job.set_micro_batch_id(micro_batch_id)
                job_list.append(recv_fwd_job)
            fw_job = core.Job(self.FORWARD + str(virtual_pp_rank))
            fw_job.set_micro_batch_id(micro_batch_id)
            job_list.append(fw_job)

        if real_split_backward:
            for micro_step in range(steady_steps):
                fwd_micro_step = micro_step + warmup_steps
                fwd_virtual_pp_rank = _get_virtual_pp_rank(
                    fwd_micro_step, forward=True
                )
                fwd_micro_batch_id = self._record_fwd_micro_step(
                    fwd_virtual_pp_rank
                )
                fwd_job = core.Job(self.FORWARD + str(fwd_virtual_pp_rank))
                fwd_job.set_micro_batch_id(fwd_micro_batch_id)
                job_list.append(fwd_job)

                bw_micro_step = micro_step
                bwd_virtual_pp_rank = _get_virtual_pp_rank(
                    bw_micro_step, forward=False
                )
                bwd_micro_batch_id = self._record_bwd_micro_step(
                    bwd_virtual_pp_rank
                )
                bwd_job = core.Job(
                    self.BACKWARD + "_b" + str(bwd_virtual_pp_rank)
                )
                bwd_job.set_micro_batch_id(bwd_micro_batch_id)
                job_list.append(bwd_job)
        else:
            for micro_step in range(steady_steps):
                fwd_micro_step = micro_step + warmup_steps
                fwd_virtual_pp_rank = _get_virtual_pp_rank(
                    fwd_micro_step, forward=True
                )
                fwd_micro_batch_id = self._record_fwd_micro_step(
                    fwd_virtual_pp_rank
                )
                bw_micro_step = micro_step
                bwd_virtual_pp_rank = _get_virtual_pp_rank(
                    bw_micro_step, forward=False
                )
                bwd_micro_batch_id = self._record_bwd_micro_step(
                    bwd_virtual_pp_rank
                )
                for job_type in self.jobs_in_stable_phase_in_pir:
                    if job_type.startswith(self.FORWARD) or job_type.startswith(
                        self.RECV_FORWARD
                    ):
                        job = core.Job(job_type + str(fwd_virtual_pp_rank))
                        job.set_micro_batch_id(fwd_micro_batch_id)
                    else:
                        job = core.Job(job_type + str(bwd_virtual_pp_rank))
                        job.set_micro_batch_id(bwd_micro_batch_id)
                    job_list.append(job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            if real_split_backward:
                bwd_job = core.Job(self.BACKWARD + "_b" + str(virtual_pp_rank))
                bwd_job.set_micro_batch_id(micro_batch_id)
                job_list.append(bwd_job)
            else:
                bwd_job = core.Job(self.BACKWARD + str(virtual_pp_rank))
                send_bwd_job = core.Job(
                    self.SEND_BACKWARD + str(virtual_pp_rank)
                )
                bwd_job.set_micro_batch_id(micro_batch_id)
                send_bwd_job.set_micro_batch_id(micro_batch_id)
                job_list.append(bwd_job)
                job_list.append(send_bwd_job)
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
                            self.BACKWARD
                            + "_w"
                            + str(chunk_id)
                            + self.reduce_comm_suffix
                        )
                    else:
                        w_job = core.Job(self.BACKWARD + "_w" + str(chunk_id))
                    w_job.set_micro_batch_id(micro_batch_id)
                    job_list.append(w_job)
        job_types = [job.type() for job in job_list]
        logger.debug(f"The VPP job list: {job_types}")
        opt_job = core.Job(self.OPT)
        job_list.append(opt_job)
        return job_list

    def _pir_split_matmul_grad_ops_to_matmul(self, program):
        for block in program.blocks:
            matmul_grad_op_idx = []
            ops = block.ops
            for i, op_i in enumerate(ops):
                if (
                    op_i.name() == "pd_op.matmul_grad"
                    and not op_i.has_attr("trans_x")
                    and not op_i.has_attr("trans_y")
                ):
                    matmul_grad_op_idx.append(i)

            for matmul_grad_id in reversed(matmul_grad_op_idx):
                _pir_split_matmul_grad_to_matmul(block, matmul_grad_id)

    def _partial_programs(self, program):
        raise RuntimeError("Not support old IR for VPP")

    def _partial_pir_programs(self, program):
        num_model_chunks = self.get_attr("vpp_degree")
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        split_backward = self.get_attr("split_backward", False)
        accumulate_steps = self.get_attr("num_micro_batches")
        num_stages = self.get_attr("pp_degree")

        if accumulate_steps != num_stages:
            split_backward = False

        assert not enable_send_recv_overlap, (
            "PIR does not support VPP with enable_send_recv_overlap yet."
        )

        if split_backward:
            self._pir_split_matmul_grad_ops_to_matmul(program)

        types, sub_program_list = self._pir_program_for_vpp(
            program, num_model_chunks, split_backward, enable_send_recv_overlap
        )

        for i in range(len(types)):
            logger.debug(
                f"type = {types[i]}, sub_programs = {sub_program_list[i]}\n"
            )

        return types, sub_program_list

    def _pir_program_for_vpp(
        self,
        program,
        num_model_chunks,
        split_bw=False,
        enable_send_recv_overlap=False,
    ):
        _pir_overlap_send_recv(program)

        oprole_names = [
            "recv_forward",
            "forward",
            "backward",
            "send_backward",
            "optimizer",
        ]
        if split_bw:
            oprole_names = ["forward", "backward_b", "backward_w", "optimizer"]

        program_types, programs = self._split_program_for_vpp(
            program, num_model_chunks, oprole_names, split_bw=split_bw
        )
        return program_types, programs

    def _split_program_for_vpp(
        self, program, num_model_chunks, oprole_names, split_bw=False
    ):
        place = _get_device()
        if isinstance(place, paddle.framework.CUDAPlace):
            place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )
        cur_place = paddle.base.libpaddle.Place()
        cur_place.set_place(place)

        def get_var_name(op_idx, result_idx):
            result_value = all_ops[op_idx].result(result_idx)
            all_used_ops = result_value.all_used_ops()
            shadow_output_op_used = None
            for op in all_used_ops:
                if op.name() == "builtin.shadow_output":
                    shadow_output_op_used = op

            if shadow_output_op_used is not None:
                var_name = shadow_output_op_used.attrs()["output_name"]
            else:
                var_name = f"var_{op_idx}_{all_ops[op_idx].name()}_{result_idx}"
            return var_name

        def add_persistable_var(op_idx, program_type):
            all_program_types = list(type_to_program.keys())
            following_program_types = all_program_types[
                all_program_types.index(program_type) + 1 :
            ]
            op_num_results = type_to_ops[program_type][op_idx].num_results()
            op_name = type_to_ops[program_type][op_idx].name()

            for idx in range(op_num_results):
                var_name = None
                for type in reversed(following_program_types):
                    op_result = type_to_ops[type][op_idx].result(idx)
                    if op_result.use_empty():
                        continue

                    # if this op's output is used, create the persistable
                    # var to be used in other programs.
                    if var_name is None:
                        if op_name in ["pd_op.data", "builtin.parameter"]:
                            var_name = op_result.name
                        else:
                            var_name = get_var_name(op_idx, idx)
                            if "var_" in var_name:
                                paddle.pir.set_insertion_point_after(
                                    type_to_ops[program_type][op_idx]
                                )
                                paddle._C_ops.set_persistable_value(
                                    type_to_ops[program_type][op_idx].result(
                                        idx
                                    ),
                                    var_name,
                                )

                    self._add_dependency_if_necessary(
                        type_to_ops, program_type, type, op_idx, idx, var_name
                    )

                    program_block = type_to_program[type].global_block()
                    new_result_var = program_block.add_kwarg(
                        var_name, op_result.type()
                    )
                    new_result_var.place_attr = cur_place
                    new_result_var.persistable = op_result.persistable
                    type_to_ops[type][op_idx].result(idx).replace_all_uses_with(
                        new_result_var
                    )

            for type in following_program_types:
                type_to_ops[type][op_idx].erase()

        type_to_program = OrderedDict()
        type_to_ops = OrderedDict()

        # Step1: create programs and ops for each type
        if not split_bw:
            chunk_ids = list(range(num_model_chunks))

            # Forward process
            for chunk_id in chunk_ids:
                for job_type in ["recv_forward", "forward"]:
                    name, prog, ops = _create_program_and_ops(
                        program, job_type, chunk_id
                    )
                    type_to_program[name] = prog
                    type_to_ops[name] = ops

            # Backward process
            for chunk_id in reversed(chunk_ids):
                for job_type in ["backward", "send_backward"]:
                    name, prog, ops = _create_program_and_ops(
                        program, job_type, chunk_id
                    )
                    type_to_program[name] = prog
                    type_to_ops[name] = ops

            # Optimizer
            name, prog, ops = _create_program_and_ops(program, "optimizer")
            type_to_program[name] = prog
            type_to_ops[name] = ops
        else:
            for type in oprole_names:
                if type == "optimizer":
                    type_to_program["optimizer"] = program.clone()
                    type_to_ops["optimizer"] = (
                        type_to_program["optimizer"].global_block().ops
                    )
                else:
                    chunk_ids = list(range(num_model_chunks))
                    if "backward" in type:
                        chunk_ids.reverse()
                    for chunk_id in chunk_ids:
                        type_to_program[type + str(chunk_id)] = program.clone()
                        type_to_ops[type + str(chunk_id)] = (
                            type_to_program[type + str(chunk_id)]
                            .global_block()
                            .ops
                        )

        # Step2: delete the ops not belong to the type
        # 1. delete ops
        # 2. add persistable var used between multiple programs
        all_ops = program.global_block().ops
        chunk_ids = list(range(num_model_chunks))
        bwd_pattern_ops_type = []

        for idx in range(len(all_ops) - 1, -1, -1):
            op = all_ops[idx]
            op_role = op.op_role
            op_chunk_id = op.chunk_id
            # Step2.1: infer chunk_id for ops that don't have chunk_id
            if op_role != int(OpRole.Optimize) and op_chunk_id == -1:
                op_chunk_id = infer_chunk_id(idx, all_ops, False)
                if op_chunk_id == -1:
                    raise ValueError(
                        f"Cannot infer chunk_id for op {op.name()} at index {idx}"
                    )

            # Step2.2: identify the job_type of the op
            if op_role == int(OpRole.Optimize):
                job_type = "optimizer"
            elif op_role == int(OpRole.Backward) and split_bw:
                if len(bwd_pattern_ops_type) == 0:
                    bwd_pattern_ops_type = _pir_get_backward_op_type(
                        all_ops, idx
                    )
                job_type = bwd_pattern_ops_type.pop()
            elif op_role == int(OpRole.Backward) and (not split_bw):
                if op.name() == "pd_op.send_v2":
                    job_type = "send_backward"
                else:
                    job_type = "backward"
            elif op_role == int(OpRole.Forward):
                if op.name() == "pd_op.recv_v2" and (not split_bw):
                    job_type = "recv_forward"
                else:
                    job_type = "forward"
            else:
                raise ValueError(
                    f"The op[{op.name()}]'s op role: {op_role} isn't one of recv_forward, forward, backward, send_backward or Optimizer."
                )

            # Step2.3: delete ops not belong to the type
            if not split_bw:
                current_type = (
                    job_type
                    if job_type == "optimizer"
                    else job_type + str(op_chunk_id)
                )

                # Get the position of the current type in type_to_program
                all_types = list(type_to_ops.keys())
                current_idx = all_types.index(current_type)

                # Delete all ops before the current type
                for type_name in all_types[:current_idx]:
                    type_to_ops[type_name][idx].erase()
            else:
                for type in oprole_names:
                    if type == job_type:
                        break
                    if type != "optimizer":
                        for chunk_id in chunk_ids:
                            type_to_ops[type + str(chunk_id)][idx].erase()
                    else:
                        type_to_ops[type][idx].erase()

                chunk_order = range(0, op_chunk_id)
                if "backward" in job_type:
                    chunk_order = range(num_model_chunks - 1, op_chunk_id, -1)
                for chunk_id in chunk_order:
                    type_to_ops[job_type + str(chunk_id)][idx].erase()

            # Step2.4: add persistable var used between multiple programs
            if job_type != "optimizer":
                add_persistable_var(idx, job_type + str(op_chunk_id))

        return list(type_to_program.keys()), list(type_to_program.values())
