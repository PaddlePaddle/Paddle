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
from ..pass_utils import _program_for_zero_bubble
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_ZBH1")
class PipelineZeroBubblePipelinePass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self.jobs_in_stable_phase = [BACKWARD + '_b', BACKWARD + '_w', FORWARD]
        self.set_attr("enable_optimizer_post_validation", 0)

    def _create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than or equal to pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_zero_bubble = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for _ in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for _ in range(micro_batch_in_zero_bubble):
            for job_type in self.jobs_in_stable_phase:
                job = core.Job(job_type)
                micro_batch_id = (
                    forward_micro_batch_id
                    if job_type.startswith(FORWARD)
                    else backward_micro_batch_id
                )
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
            forward_micro_batch_id += 1
            backward_micro_batch_id += 1

        for _ in range(micro_batch_in_warmup):
            backward_b_job = core.Job(BACKWARD + '_b')
            backward_b_job.set_micro_batch_id(backward_micro_batch_id)

            backward_w_job = core.Job(BACKWARD + '_w')
            backward_w_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_b_job)

            backward_micro_batch_id += 1

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        types, sub_program_list = _program_for_zero_bubble(
            program, dist_context
        )
        return types, sub_program_list
