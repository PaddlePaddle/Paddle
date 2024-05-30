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

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    _program_for_vpp,
    _program_for_vpp_split_bwk,
    _split_and_replace_recv,
    shadow_var_between_sub_programs,
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

        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}
        self._types = []

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
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        stage_id = self.get_attr("pp_stage")
        num_stages = self.get_attr("pp_degree")
        num_model_chunks = self.get_attr("vpp_degree")
        split_backward = self.get_attr("split_backward", False)
        print("====stage_id: ", stage_id, flush=1)
        print("====num_stages: ", num_stages, flush=1)
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
            fw_job_name = FORWARD + str(virtual_pp_rank)
            if (
                enable_send_recv_overlap
                and ("recv4_" + fw_job_name) in self._types
            ):
                fwd_event = "vpp_warmup_fwd_" + str(micro_step)
                recv4_fw_job = core.Job("recv4_" + fw_job_name)
                recv4_fw_job.set_micro_batch_id(micro_batch_id)
                recv4_fw_job.set_event_to_record(fwd_event)
                job_list.append(recv4_fw_job)

                fw_job = core.Job("no_recv_" + fw_job_name)
                fw_job.set_micro_batch_id(micro_batch_id)
                fw_job.set_event_to_wait(fwd_event)
                job_list.append(fw_job)
            else:
                fw_job = core.Job(fw_job_name)
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
            fwd_job_name = FORWARD + str(fwd_virtual_pp_rank)
            if (
                enable_send_recv_overlap
                and ("recv4_" + fwd_job_name) in self._types
                and len(job_list) > 0
            ):
                fwd_event = "vpp_steady_fwd_" + str(fwd_micro_step)
                recv4_fwd_job = core.Job("recv4_" + fwd_job_name)
                recv4_fwd_job.set_micro_batch_id(fwd_micro_batch_id)
                recv4_fwd_job.set_event_to_record(fwd_event)
                print("add event: ", fwd_event, flush=1)
                if micro_step > 0:
                    job_list.insert(-1, recv4_fwd_job)
                else:
                    job_list.append(recv4_fwd_job)

                fwd_job = core.Job("no_recv_" + fwd_job_name)
                fwd_job.set_micro_batch_id(fwd_micro_batch_id)
                fwd_job.set_event_to_wait(fwd_event)
                job_list.append(fwd_job)
            else:
                fwd_job = core.Job(fwd_job_name)
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
                bwd_job_name = BACKWARD + "_b" + str(bwd_virtual_pp_rank)
            else:
                bwd_job_name = BACKWARD + str(bwd_virtual_pp_rank)
            if (
                enable_send_recv_overlap
                and ("recv4_" + bwd_job_name) in self._types
                and len(job_list) > 0
            ):
                job_name = "recv4_" + bwd_job_name
                if stage_id == 0 and micro_step == 0:
                    job_name = job_name + "_with_sync"
                print("======>recv_overlap for ", micro_step, flush=1)
                bwd_event = "vpp_steady_bwd_" + str(bw_micro_step)
                recv4_bwd_job = core.Job(job_name)
                recv4_bwd_job.set_micro_batch_id(bwd_micro_batch_id)
                recv4_bwd_job.set_event_to_record(bwd_event)
                print("add event: ", bwd_event, flush=1)
                job_list.insert(-1, recv4_bwd_job)

                bwd_job = core.Job("no_recv_" + bwd_job_name)
                bwd_job.set_micro_batch_id(bwd_micro_batch_id)
                bwd_job.set_event_to_wait(bwd_event)
                job_list.append(bwd_job)
            else:
                bwd_job = core.Job(bwd_job_name)
                bwd_job.set_micro_batch_id(bwd_micro_batch_id)
                job_list.append(bwd_job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            if real_split_backward:
                bwd_job_name = BACKWARD + "_b" + str(virtual_pp_rank)
            else:
                bwd_job_name = BACKWARD + str(virtual_pp_rank)
            if (
                enable_send_recv_overlap
                and ("recv4_" + bwd_job_name) in self._types
            ):
                bwd_event = "vpp_cooldowm_bwd_" + str(micro_step)
                recv4_bwd_job = core.Job("recv4_" + bwd_job_name)
                recv4_bwd_job.set_micro_batch_id(micro_batch_id)
                recv4_bwd_job.set_event_to_record(bwd_event)
                job_list.append(recv4_bwd_job)

                bwd_job = core.Job("no_recv_" + bwd_job_name)
                bwd_job.set_micro_batch_id(micro_batch_id)
                bwd_job.set_event_to_wait(bwd_event)
                job_list.append(bwd_job)
            else:
                bwd_job = core.Job(bwd_job_name)
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
                    w_job_name = BACKWARD + "_w" + str(chunk_id)
                    if (
                        enable_send_recv_overlap
                        and ("recv4_" + bwd_job_name) in self._types
                    ):
                        bwd_event = "vpp_cooldowm_bwd_w_" + str(micro_batch_id)
                        recv4_w_job = core.Job("recv4_" + w_job_name)
                        recv4_w_job.set_micro_batch_id(micro_batch_id)
                        recv4_w_job.set_event_to_record(bwd_event)
                        job_list.append(recv4_w_job)
                        w_job = core.Job("no_recv_" + w_job_name)
                        w_job.set_micro_batch_id(micro_batch_id)
                        w_job.set_event_to_wait(bwd_event)
                        job_list.append(w_job)
                    else:
                        w_job = core.Job(w_job_name)
                        w_job.set_micro_batch_id(micro_batch_id)
                        job_list.append(w_job)

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

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        num_model_chunks = self.get_attr("vpp_degree")
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        accumulate_steps = self.get_attr("num_micro_batches")
        num_stages = self.get_attr("pp_degree")
        split_backward = self.get_attr("split_backward", False)

        if split_backward and accumulate_steps == num_stages:
            self._split_matmul_grad_ops_to_matmul(program, dist_context)
            types, sub_program_list = _program_for_vpp_split_bwk(
                program,
                num_model_chunks,
                dist_context,
                enable_send_recv_overlap,
            )
        else:
            types, sub_program_list = _program_for_vpp(
                program,
                num_model_chunks,
                dist_context,
                enable_send_recv_overlap,
            )

        enable_pir_in_executor = paddle.framework.get_flags(
            "FLAGS_enable_pir_in_executor"
        )['FLAGS_enable_pir_in_executor']
        if enable_pir_in_executor:
            shadow_var_between_sub_programs(sub_program_list)

        if enable_send_recv_overlap:
            types, sub_program_list = _split_and_replace_recv(
                types, sub_program_list, self.get_attr("vpp_degree")
            )
        self._types = types
        return types, sub_program_list
