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

import paddle
from paddle.base import core
from paddle.framework import (
    _current_expected_place_ as _get_device,
)

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    AutoParallelStreamType,
    forward_complete_op_role,
    split_program,
)
from .pipeline_pass_base import PipelinePassBase

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_1F1B")
class Pipeline1F1BPass(PipelinePassBase):

    def __init__(self):
        super().__init__()
        self.jobs_in_stable_phase = [self.BACKWARD, self.FORWARD]
        self.jobs_in_stable_phase_in_pir = [
            self.BACKWARD,
            self.RECV_FORWARD,
            self.SEND_BACKWARD,
            self.FORWARD,
        ]
        self.set_attr("enable_backward_forward_overlap", 0)

    def _create_job_list(self):
        if self._in_pir_mode:
            return self._create_job_list_in_pir()
        else:
            raise NotImplementedError(
                "_create_job_list() only support PIR now."
            )

    def _create_job_list_in_pir(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than or equal to pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for i in range(micro_batch_in_warmup):
            recv_fwd_job = core.Job(self.RECV_FORWARD)
            recv_fwd_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(recv_fwd_job)

            forward_job = core.Job(self.FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for i in range(micro_batch_in_1f1b):
            for job_type in self.jobs_in_stable_phase_in_pir:
                job = core.Job(job_type)
                micro_batch_id = (
                    forward_micro_batch_id
                    if job_type.startswith(self.FORWARD)
                    or job_type.startswith(self.RECV_FORWARD)
                    else backward_micro_batch_id
                )
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
            forward_micro_batch_id += 1
            backward_micro_batch_id += 1

        for i in range(micro_batch_in_warmup):
            backward_job = core.Job(self.BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)

            send_bwd_job = core.Job(self.SEND_BACKWARD)
            send_bwd_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(send_bwd_job)

            backward_micro_batch_id += 1

        opt_job = core.Job(self.OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        raise NotImplementedError("pipeline_1f1b_pass() only support PIR now.")

    def _partial_pir_programs(self, program):
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        assert (
            not enable_send_recv_overlap
        ), "PIR does not support 1F1B with enable_send_recv_overlap yet."

        self._overlap_send_recv(program)
        forward_complete_op_role(program)

        job_types = [
            self.RECV_FORWARD,
            self.FORWARD,
            self.BACKWARD,
            self.SEND_BACKWARD,
            self.OPT,
        ]

        programs = {}
        for job_type in job_types:
            programs[job_type] = program.clone()

        complete_ops = program.global_block().ops
        ops_dict = {
            key: prog.global_block().ops for key, prog in programs.items()
        }
        blocks_dict = {
            key: prog.global_block() for key, prog in programs.items()
        }

        region = "opt"
        for op_idx in range(len(complete_ops) - 1, -1, -1):
            op = complete_ops[op_idx]
            if op.op_role != -1:
                if op.op_role == 1:
                    region = "bwd"
                elif op.op_role == 0:
                    region = "fwd"
                elif op.op_role == 2:
                    region = "opt"

            if region == "opt":
                self._erase_op_from_other_programs(
                    op_idx, self.OPT, ops_dict, job_types
                )
            elif region == "bwd" and op.name() == "pd_op.send_v2":
                self._handle_func(
                    op_idx,
                    self.SEND_BACKWARD,
                    job_types[4:],
                    complete_ops,
                    ops_dict,
                    blocks_dict,
                )
                self._erase_op_from_other_programs(
                    op_idx, self.SEND_BACKWARD, ops_dict, job_types
                )
            elif region == "bwd" and op.name() != "pd_op.send_v2":
                self._handle_func(
                    op_idx,
                    self.BACKWARD,
                    job_types[3:],
                    complete_ops,
                    ops_dict,
                    blocks_dict,
                )
                self._erase_op_from_other_programs(
                    op_idx, self.BACKWARD, ops_dict, job_types
                )
            elif region == "fwd" and op.name() != "pd_op.recv_v2":
                self._handle_func(
                    op_idx,
                    self.FORWARD,
                    job_types[2:],
                    complete_ops,
                    ops_dict,
                    blocks_dict,
                )
                self._erase_op_from_other_programs(
                    op_idx, self.FORWARD, ops_dict, job_types
                )
            elif region == "fwd" and op.name() == "pd_op.recv_v2":
                self._handle_func(
                    op_idx,
                    self.RECV_FORWARD,
                    job_types[1:],
                    complete_ops,
                    ops_dict,
                    blocks_dict,
                )
                self._erase_op_from_other_programs(
                    op_idx, self.RECV_FORWARD, ops_dict, job_types
                )
        sub_program_list = []
        for job_type in job_types:
            sub_program_list.append(programs[job_type])
        for i in range(len(job_types)):
            logger.debug(
                f"type = {job_types[i]}, sub_programs = {sub_program_list[i]}\n"
            )
        logger.debug(
            f"jobs_in_stable_phase = {self.jobs_in_stable_phase_in_pir}"
        )
        return job_types, sub_program_list

    def _split_program_for_overlapping(self, job_type, program, split_points):
        assert job_type in [
            self.FORWARD,
            self.BACKWARD,
        ], f"job_type should be one of {[self.FORWARD, self.BACKWARD]}"

        split_programs, __, __ = split_program(program, split_points)

        split_job_types = []
        num_split_programs = len(split_programs)
        for idx in range(num_split_programs):
            split_job_types.append(f"{job_type}(chunk{idx})")

        return split_job_types, split_programs

    def is_comm_op_valid_to_overlap(self, op):
        return (
            op.type == "all_reduce"
            and op.attr("reduce_type") == paddle.distributed.ReduceOp.SUM
            and op.dist_attr.execution_stream
            == AutoParallelStreamType.CALC_STREAM.value
        )

    def _handle_func(
        self,
        op_idx,
        cur_job_type,
        suffixed_job_types,
        complete_ops,
        ops_dict,
        blocks_dict,
    ):
        for idx in range(complete_ops[op_idx].num_results()):
            if self._result_is_used(suffixed_job_types, op_idx, idx, ops_dict):
                var_name = self._get_or_create_var_name(
                    ops_dict[cur_job_type], op_idx, idx, complete_ops
                )

            for job_type in suffixed_job_types:
                if self._result_is_used([job_type], op_idx, idx, ops_dict):
                    self._add_dependency_if_necessary(
                        ops_dict, cur_job_type, job_type, op_idx, idx, var_name
                    )
                    self._add_kwarg_and_replace(
                        blocks_dict[job_type],
                        ops_dict[job_type],
                        op_idx,
                        idx,
                        var_name,
                    )

    def _result_is_used(self, job_types, op_idx, rst_idx, ops_dict):
        is_used = False
        for job_type in job_types:
            is_used = (
                is_used
                or ops_dict[job_type][op_idx].result(rst_idx).use_empty()
                is False
            )
        return is_used

    def _get_or_create_var_name(
        self, cur_sub_ops, op_idx, rst_idx, complete_ops
    ):
        var_name = None
        # case1: get var_name in current sub-program
        op = cur_sub_ops[op_idx]
        if op.name() == "pd_op.data" or op.name() == "builtin.parameter":
            var_name = op.result(rst_idx).name
        else:
            # case2: get var_name from shadow_output in complete program
            result_var = complete_ops[op_idx].result(rst_idx)
            shadow_output_op = None
            for used_op in result_var.all_used_ops():
                if used_op.name() == "builtin.shadow_output":
                    shadow_output_op = used_op
            if shadow_output_op is not None:
                var_name = shadow_output_op.attrs()["output_name"]

        if var_name is None:
            # case3: create var_name in current sub-program
            paddle.pir.set_insertion_point_after(op)
            var_name = f"var_{op_idx}_{complete_ops[op_idx].name()}_{rst_idx}"
            paddle._C_ops.set_persistable_value(op.result(rst_idx), var_name)
        return var_name

    def _add_kwarg_and_replace(self, block, ops, op_idx, rst_idx, var_name):
        ori_result = ops[op_idx].result(rst_idx)
        new_result_var = block.add_kwarg(var_name, ori_result.type())
        new_result_var.place_attr = self._get_cur_place()
        new_result_var.persistable = ori_result.persistable
        ops[op_idx].result(rst_idx).replace_all_uses_with(new_result_var)

    def _overlap_send_recv(self, program):
        for block in program.blocks:
            for op in block.ops:
                if op.name() == "pd_op.send_v2":
                    op.set_bool_attr("dynamic_shape", False)
                    op.set_bool_attr("use_calc_stream", True)
                    ring_id = op.attrs()["ring_id"]
                    op.set_execution_stream("send_recv_stream")
                    op.set_scheduling_priority(0)
                elif op.name() == "pd_op.recv_v2":
                    op.set_bool_attr("dynamic_shape", False)
                    op.set_bool_attr("use_calc_stream", True)
                    op.set_execution_stream("send_recv_stream")
                    op.set_scheduling_priority(0)

    def _erase_op_from_other_programs(
        self, op_idx, keep_job_type, ops_dict, job_types
    ):
        for job_type in job_types:
            if job_type != keep_job_type:
                ops_dict[job_type][op_idx].erase()

    def _get_cur_place(self):
        place = _get_device()
        if isinstance(place, paddle.framework.CUDAPlace):
            place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )
        cur_place = paddle.base.libpaddle.Place()
        cur_place.set_place(place)
        return cur_place
