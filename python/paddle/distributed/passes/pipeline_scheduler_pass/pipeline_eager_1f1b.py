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
from ..pass_utils import _program_for_fthenb_and_1f1b
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_Eager1F1B")
class PipelineEager1F1BPass(PipelinePassBase):
    def __init__(self):
        super().__init__()

    def _create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        assert (
            2 * (pp_degree - pp_stage) - 1 <= num_micro_batches
        ), "Num of micro batches should larger than 2 * (pp_degree - pp_stage) - 1."

        micro_batch_in_warmup = 2 * (pp_degree - pp_stage) - 1
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for _ in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for _ in range(micro_batch_in_1f1b):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        for _ in range(micro_batch_in_warmup):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1

        opt_job = core.Job(OPT)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        # NOTE: The flag "enable_send_recv_overlap" may increase the reserved memory of GPUs.
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        # TODO: More function will be added later. Now it uses the same logic as FTthenB and 1F1B.
        types = [FORWARD, BACKWARD, OPT]
        sub_program_list = _program_for_fthenb_and_1f1b(
            program, enable_send_recv_overlap
        )
        return types, sub_program_list
