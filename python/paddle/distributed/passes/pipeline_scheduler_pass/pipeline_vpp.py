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

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import _program_for_vpp
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

        job_list = []
        for micro_step in range(warmup_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=True)
            micro_batch_id = self._record_fwd_micro_step(virtual_pp_rank)
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
            bwd_job = core.Job(BACKWARD + str(bwd_virtual_pp_rank))
            bwd_job.set_micro_batch_id(bwd_micro_batch_id)
            job_list.append(bwd_job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            bwd_job = core.Job(BACKWARD + str(virtual_pp_rank))
            bwd_job.set_micro_batch_id(micro_batch_id)
            job_list.append(bwd_job)

        opt_job = core.Job(OPT)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        num_model_chunks = self.get_attr("vpp_degree")
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types, sub_program_list = _program_for_vpp(
            program, num_model_chunks, dist_context, enable_send_recv_overlap
        )
        return types, sub_program_list
